# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import pathlib
import warnings
from typing import Dict, Any, List, Union, Optional

import fsspec
import numpy as np
import xarray as xr
import zarr

from xcube.core.gridmapping import GridMapping
from xcube.core.mldataset import BaseMultiLevelDataset
from xcube.core.mldataset import LazyMultiLevelDataset
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.subsampling import AGG_METHODS
# Note, we need the following reference to register the
# xarray property accessor
# noinspection PyUnresolvedReferences
from xcube.core.zarrstore import ZarrStoreHolder
from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonComplexSchema
from xcube.util.jsonschema import JsonIntegerSchema
from xcube.util.jsonschema import JsonNullSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema
from xcube.util.types import normalize_scalar_or_pair
from .dataset import DatasetZarrFsDataAccessor
from ..helpers import get_fs_path_class
from ..helpers import resolve_path
from ...datatype import DATASET_TYPE
from ...datatype import DataType
from ...datatype import MULTI_LEVEL_DATASET_TYPE
from ...error import DataStoreError


class MultiLevelDatasetLevelsFsDataAccessor(DatasetZarrFsDataAccessor):
    """
    Opener/writer extension name "mldataset:levels:<protocol>".
    """

    @classmethod
    def get_data_type(cls) -> DataType:
        return MULTI_LEVEL_DATASET_TYPE

    @classmethod
    def get_format_id(cls) -> str:
        return 'levels'

    def open_data(self, data_id: str, **open_params) -> MultiLevelDataset:
        assert_instance(data_id, str, name='data_id')
        fs, root, open_params = self.load_fs(open_params)
        return FsMultiLevelDataset(fs, root, data_id, **open_params)

    def get_write_data_params_schema(self) -> JsonObjectSchema:
        schema = super().get_write_data_params_schema()  # creates deep copy
        # TODO: remove use_saved_levels, instead see #619
        schema.properties['use_saved_levels'] = JsonBooleanSchema(
            description='Whether to open an already saved level'
                        ' and downscale it then.'
                        ' May be used to avoid computation of'
                        ' entire Dask graphs at each level.',
            default=False,
        )
        schema.properties['base_dataset_id'] = JsonStringSchema(
            description='If given, avoids writing the base dataset'
                        ' at level 0. Instead a file "{data_id}/0.link"'
                        ' is created whose content is the given base dataset'
                        ' identifier.',
            nullable=True,
        )
        schema.properties['tile_size'] = JsonComplexSchema(
            one_of=[
                JsonIntegerSchema(minimum=1),
                JsonArraySchema(items=[
                    JsonIntegerSchema(minimum=1),
                    JsonIntegerSchema(minimum=1)
                ]),
                JsonNullSchema(),
            ],
            description='Tile size to be used for all levels of the'
                        ' written multi-level dataset.',
        )
        schema.properties['num_levels'] = JsonIntegerSchema(
            description='Maximum number of levels to be written.',
            minimum=1,
            nullable=True,
        )
        schema.properties['agg_methods'] = JsonComplexSchema(
            one_of=[
                JsonStringSchema(enum=AGG_METHODS),
                JsonObjectSchema(
                    additional_properties=JsonStringSchema(enum=AGG_METHODS)
                ),
                JsonNullSchema(),
            ],
            description='Aggregation method for the pyramid levels.'
                        ' If not explicitly set, "first" is used for integer '
                        ' variables and "mean" for floating point variables.'
                        ' If given as object, it is a mapping from variable '
                        ' name pattern to aggregation method. The pattern'
                        ' may include wildcard characters * and ?.'
        )
        return schema

    def write_data(self,
                   data: MultiLevelDataset,
                   data_id: str,
                   replace: bool = False,
                   **write_params) -> str:
        assert_instance(data, MultiLevelDataset, name='data')
        return self.write_generic_data(data,
                                       data_id,
                                       replace=replace,
                                       **write_params)

    def write_generic_data(self,
                           data: Union[xr.Dataset, MultiLevelDataset],
                           data_id: str,
                           replace: bool = False,
                           **write_params) -> str:
        assert_instance(data, (xr.Dataset, MultiLevelDataset), name='data')
        assert_instance(data_id, str, name='data_id')
        tile_size = write_params.pop('tile_size', None)
        agg_methods = write_params.pop('agg_methods', None)
        if isinstance(data, MultiLevelDataset):
            ml_dataset = data
            if tile_size:
                warnings.warn(
                    'tile_size is ignored for multi-level datasets'
                )
            if agg_methods:
                warnings.warn(
                    'agg_methods is ignored for multi-level datasets'
                )
        else:
            base_dataset: xr.Dataset = data
            grid_mapping = None
            if tile_size is not None:
                tile_size = normalize_scalar_or_pair(
                    tile_size, item_type=int, name='tile_size'
                )
                grid_mapping = GridMapping.from_dataset(base_dataset)
                x_name, y_name = grid_mapping.xy_dim_names
                # noinspection PyTypeChecker
                base_dataset = base_dataset.chunk({x_name: tile_size[0],
                                                   y_name: tile_size[1]})
                # noinspection PyTypeChecker
                grid_mapping = grid_mapping.derive(tile_size=tile_size)
            ml_dataset = BaseMultiLevelDataset(base_dataset,
                                               grid_mapping=grid_mapping,
                                               agg_methods=agg_methods)
        fs, root, write_params = self.load_fs(write_params)
        consolidated = write_params.pop('consolidated', True)
        use_saved_levels = write_params.pop('use_saved_levels', False)
        base_dataset_id = write_params.pop('base_dataset_id', None)
        num_levels = write_params.pop('num_levels', None)

        if use_saved_levels:
            ml_dataset = BaseMultiLevelDataset(
                ml_dataset.get_dataset(0),
                grid_mapping=ml_dataset.grid_mapping,
                agg_methods=agg_methods
            )

        path_class = get_fs_path_class(fs)
        data_path = path_class(data_id)
        fs.mkdirs(str(data_path), exist_ok=replace)

        if num_levels is None or num_levels <= 0:
            num_levels_max = ml_dataset.num_levels
        else:
            num_levels_max = min(num_levels, ml_dataset.num_levels)

        for index in range(num_levels_max):
            level_dataset = ml_dataset.get_dataset(index)
            if base_dataset_id and index == 0:
                # Write file "0.link" instead of copying
                # level zero dataset to "0.zarr".

                # Compute a relative base dataset path first
                base_dataset_path = path_class(root, base_dataset_id)
                data_parent_path = data_path.parent
                try:
                    base_dataset_path = base_dataset_path.relative_to(
                        data_parent_path
                    )
                except ValueError as e:
                    raise DataStoreError(
                        f'invalid base_dataset_id: {base_dataset_id}'
                    ) from e
                base_dataset_path = '..' / base_dataset_path

                # Then write relative base dataset path into link file
                link_path = data_path / f'{index}.link'
                with fs.open(str(link_path), mode='w') as fp:
                    fp.write(base_dataset_path.as_posix())
            else:
                # Write level "{index}.zarr"
                level_path = data_path / f'{index}.zarr'
                level_zarr_store = fs.get_mapper(str(level_path), create=True)
                try:
                    level_dataset.to_zarr(
                        level_zarr_store,
                        mode='w' if replace else None,
                        consolidated=consolidated,
                        **write_params
                    )
                except ValueError as e:
                    # TODO: remove already written data!
                    raise DataStoreError(f'Failed to write'
                                         f' dataset {data_id}: {e}') from e
                if use_saved_levels:
                    level_dataset = xr.open_zarr(level_zarr_store,
                                                 consolidated=consolidated)
                    level_dataset.zarr_store.set(level_zarr_store)
                    ml_dataset.set_dataset(index, level_dataset)

        return data_id


class DatasetLevelsFsDataAccessor(MultiLevelDatasetLevelsFsDataAccessor):
    """
    Opener/writer extension name "dataset:levels:<protocol>".
    """

    @classmethod
    def get_data_type(cls) -> DataType:
        return DATASET_TYPE

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        ml_dataset = super().open_data(data_id, **open_params)
        return ml_dataset.get_dataset(0)

    def write_data(self,
                   data: xr.Dataset,
                   data_id: str,
                   replace: bool = False,
                   **write_params) -> str:
        assert_instance(data, xr.Dataset, name='data')
        return self.write_generic_data(data,
                                       data_id,
                                       replace=replace,
                                       **write_params)


class FsMultiLevelDataset(LazyMultiLevelDataset):
    _MIN_CACHE_SIZE = 1024 * 1024  # 1 MiB

    def __init__(self,
                 fs: fsspec.AbstractFileSystem,
                 root: Optional[str],
                 data_id: str,
                 **open_params: Dict[str, Any]):
        super().__init__(ds_id=data_id)
        self._fs = fs
        self._root = root
        self._path = data_id
        self._open_params = open_params
        self._path_class = get_fs_path_class(fs)
        self._size_weights: Optional[np.ndarray] = None

    @property
    def size_weights(self) -> np.ndarray:
        if self._size_weights is None:
            with self.lock:
                self._size_weights = self.compute_size_weights(
                    self.num_levels
                )
        return self._size_weights

    def _get_dataset_lazily(self, index: int, parameters) \
            -> xr.Dataset:

        open_params = dict(self._open_params)

        cache_size = open_params.pop('cache_size', None)

        fs = self._fs

        ds_path = self._get_path(self._path)
        link_path = ds_path / f'{index}.link'
        if fs.isfile(str(link_path)):
            # If file "{index}.link" exists, we have a link to
            # a level Zarr and open this instead,
            with fs.open(str(link_path), 'r') as fp:
                level_path = self._get_path(fp.read())
            if not level_path.is_absolute() \
                    and not level_path.is_relative_to(ds_path):
                level_path = resolve_path(ds_path / level_path)
        else:
            # Nominal "{index}.zarr" must exist
            level_path = ds_path / f'{index}.zarr'

        level_zarr_store = fs.get_mapper(str(level_path))

        consolidated = open_params.pop('consolidated',
                                       '.zmetadata' in level_zarr_store)

        if isinstance(cache_size, int) \
                and cache_size >= self._MIN_CACHE_SIZE:
            # compute cache size for level weighted by
            # size in pixels for each level
            cache_size = math.ceil(self.size_weights[index] * cache_size)
            if cache_size >= self._MIN_CACHE_SIZE:
                level_zarr_store = zarr.LRUStoreCache(level_zarr_store,
                                                      max_size=cache_size)

        try:
            level_dataset = xr.open_zarr(level_zarr_store,
                                         consolidated=consolidated,
                                         **open_params)
        except ValueError as e:
            raise DataStoreError(f'Failed to open'
                                 f' dataset {level_path!r}:'
                                 f' {e}') from e

        level_dataset.zarr_store.set(level_zarr_store)
        return level_dataset

    @staticmethod
    def compute_size_weights(num_levels: int) -> np.ndarray:
        weights = (2 ** np.arange(0, num_levels, dtype=np.float64)) ** 2
        return weights[::-1] / np.sum(weights)

    def _get_num_levels_lazily(self) -> int:
        levels = self._get_levels()
        num_levels = len(levels)
        expected_levels = list(range(num_levels))
        for level in expected_levels:
            if level != levels[level]:
                raise DataStoreError(f'Inconsistent'
                                     f' multi-level dataset {self.ds_id!r},'
                                     f' expected levels {expected_levels!r}'
                                     f' found {levels!r}')
        return num_levels

    def _get_levels(self) -> List[int]:
        levels = []
        paths = [self._get_path(entry['name'])
                 for entry in self._fs.listdir(self._path, detail=True)]
        for path in paths:
            # No ext, i.e. dir_name = "<level>", is proposed by
            # https://github.com/zarr-developers/zarr-specs/issues/50.
            # xcube already selected dir_name = "<level>.zarr".
            basename = path.stem
            if path.stem and path.suffix in ('', '.zarr', '.link'):
                try:
                    level = int(basename)
                except ValueError:
                    continue
                levels.append(level)
        levels = sorted(levels)
        return levels

    def _get_path(self, *args) -> pathlib.PurePath:
        return self._path_class(*args)
