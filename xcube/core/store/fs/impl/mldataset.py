# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
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

import os.path
from typing import Dict, Any, List, Union, Tuple, Optional

import fsspec
import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.core.mldataset import BaseMultiLevelDataset
from xcube.core.mldataset import LazyMultiLevelDataset
from xcube.core.mldataset import MultiLevelDataset
from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema
from xcube.util.jsonschema import JsonIntegerSchema
from xcube.util.tilegrid import TileGrid
from .dataset import DatasetZarrFsDataAccessor
from ...datatype import DATASET_TYPE
from ...datatype import DataType
from ...datatype import MULTI_LEVEL_DATASET_TYPE
from ...error import DataStoreError


class MultiLevelDatasetLevelsFsDataAccessor(DatasetZarrFsDataAccessor):
    """
    Opener/writer extension name: "mldataset:levels:<protocol>"
    and "dataset:levels:<protocol>"
    """

    @classmethod
    def get_data_types(cls) -> Tuple[DataType, ...]:
        return MULTI_LEVEL_DATASET_TYPE, DATASET_TYPE

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
        )
        schema.properties['tile_size'] = JsonIntegerSchema(
            description='Tile size to be used for all levels of the'
                        ' written multi-level dataset.',
        )
        return schema

    def write_data(self,
                   data: Union[xr.Dataset, MultiLevelDataset],
                   data_id: str,
                   replace: bool = False,
                   **write_params) -> str:
        assert_instance(data, (xr.Dataset, MultiLevelDataset), name='data')
        assert_instance(data_id, str, name='data_id')
        if isinstance(data, MultiLevelDataset):
            ml_dataset = data
        else:
            base_dataset: xr.Dataset = data
            tile_size = write_params.pop('tile_size', None)
            if tile_size:
                assert_instance(tile_size, int, name='tile_size')
                gm = GridMapping.from_dataset(base_dataset)
                x_name, y_name = gm.xy_dim_names
                base_dataset = base_dataset.chunk({x_name: tile_size,
                                                   y_name: tile_size})
            ml_dataset = BaseMultiLevelDataset(base_dataset)
        fs, root, write_params = self.load_fs(write_params)
        consolidated = write_params.pop('consolidated', True)
        use_saved_levels = write_params.pop('use_saved_levels', False)
        base_dataset_id = write_params.pop('base_dataset_id', None)

        if use_saved_levels:
            ml_dataset = BaseMultiLevelDataset(
                ml_dataset.get_dataset(0),
                tile_grid=ml_dataset.tile_grid
            )

        for index in range(ml_dataset.num_levels):
            level_dataset = ml_dataset.get_dataset(index)
            if base_dataset_id and index == 0:
                # Write file "0.link" instead of copying
                # level-0 dataset to "0.zarr"
                link_path = f'{data_id}/{index}.link'
                fs.mkdirs(data_id, exist_ok=replace)
                with fs.open(link_path, mode='w') as fp:
                    fp.write(base_dataset_id)
            else:
                # Write level "{index}.zarr"
                level_path = f'{data_id}/{index}.zarr'
                zarr_store = fs.get_mapper(level_path, create=True)
                try:
                    level_dataset.to_zarr(
                        zarr_store,
                        mode='w' if replace else None,
                        consolidated=consolidated,
                        **write_params
                    )
                except ValueError as e:
                    # TODO: remove already written data!
                    raise DataStoreError(f'Failed to write'
                                         f' dataset {data_id}: {e}') from e
                if use_saved_levels:
                    level_dataset = xr.open_zarr(zarr_store,
                                                 consolidated=consolidated)
                    ml_dataset.set_dataset(index, level_dataset)

        return data_id



class FsMultiLevelDataset(LazyMultiLevelDataset):
    def __init__(self,
                 fs: fsspec.AbstractFileSystem,
                 root: Optional[str],
                 data_id: str,
                 **open_params: Dict[str, Any]):
        super().__init__(ds_id=data_id)
        self._fs = fs
        self._root = root
        self._num_levels = self._get_num_levels(fs, data_id)
        self._open_params = open_params

    def _get_dataset_lazily(self, index: int, parameters) \
            -> xr.Dataset:

        open_params = dict(self._open_params)

        link_path = f'{self.ds_id}/{index}.link'
        if self._fs.isfile(link_path):
            # If file "{index}.link" exists, we have a link to
            # a level Zarr and open this instead,
            with self._fs.open(link_path, 'r') as fp:
                level_path = fp.read()
            if self._root:
                # If we have a root and level_path is relative
                # make it absolute using root
                is_abs = os.path.isabs(level_path) or '://' in level_path
                if not is_abs:
                    level_path = f'{self._root}/{level_path}'
        else:
            # Nominal "{index}.zarr" must exist
            level_path = f'{self.ds_id}/{index}.zarr'

        consolidated = open_params.pop(
            'consolidated',
            self._fs.exists(f'{level_path}/.zmetadata')
        )

        level_zarr_store = self._fs.get_mapper(level_path)
        try:
            return xr.open_zarr(level_zarr_store,
                                consolidated=consolidated,
                                **open_params)
        except ValueError as e:
            raise DataStoreError(f'Failed to open'
                                 f' dataset {level_path!r}:'
                                 f' {e}') from e

    def _get_tile_grid_lazily(self) -> TileGrid:
        """
        Retrieve, i.e. read or compute, the tile grid
        used by the multi-level dataset.

        :return: the dataset for the level at *index*.
        """
        tile_grid = self.grid_mapping.tile_grid
        if tile_grid.num_levels != self._num_levels:
            raise DataStoreError(f'Detected inconsistent'
                                 f' number of detail levels,'
                                 f' expected {tile_grid.num_levels},'
                                 f' found {self._num_levels}.')
        return tile_grid

    @classmethod
    def _get_num_levels(cls,
                        fs: fsspec.AbstractFileSystem,
                        data_id: str) -> int:
        levels = cls._get_levels(fs, data_id)
        num_levels = len(levels)
        expected_levels = list(range(num_levels))
        for level in expected_levels:
            if level != levels[level]:
                raise DataStoreError(f'Inconsistent'
                                     f' multi-level dataset {data_id!r},'
                                     f' expected levels {expected_levels!r}'
                                     f' found {levels!r}')
        return num_levels

    @classmethod
    def _get_levels(cls,
                    fs: fsspec.AbstractFileSystem,
                    data_id: str) -> List[int]:
        levels = []
        for dir_name in (os.path.basename(dir_path['name'])
                         for dir_path in fs.listdir(data_id, detail=True)):
            # No ext, i.e. dir_name = "<level>", is proposed by
            # https://github.com/zarr-developers/zarr-specs/issues/50.
            # xcube already selected dir_name = "<level>.zarr".
            basename, ext = os.path.splitext(dir_name)
            if basename and ext in ('', '.zarr', '.link'):
                try:
                    level = int(basename)
                except ValueError:
                    continue
                levels.append(level)
            try:
                level = int(dir_name)
            except ValueError:
                continue
            levels.append(level)
        levels = sorted(levels)
        return levels
