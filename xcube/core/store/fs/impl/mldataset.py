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
from typing import Dict, Any, List, Union, Tuple

import fsspec
import xarray as xr

from xcube.core.mldataset import BaseMultiLevelDataset
from xcube.core.mldataset import LazyMultiLevelDataset
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.mldataset import get_dataset_tile_grid
from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonObjectSchema
from .dataset import DatasetZarrFsDataAccessor
from ...datatype import DATASET_TYPE
from ...datatype import DataType
from ...datatype import MULTI_LEVEL_DATASET_TYPE
from ...error import DataStoreError


class MultiLevelDatasetLevelsFsDataAccessor(DatasetZarrFsDataAccessor):
    """
    Opener/writer extension name: "mldataset:levels:<fs_protocol>"
    and "dataset:levels:<fs_protocol>"
    """

    @classmethod
    def get_data_types(cls) -> Tuple[DataType, ...]:
        return MULTI_LEVEL_DATASET_TYPE, DATASET_TYPE

    @classmethod
    def get_format_id(cls) -> str:
        return 'levels'

    def open_data(self, data_id: str, **open_params) -> MultiLevelDataset:
        assert_instance(data_id, str, name='data_id')
        fs, open_params = self.load_fs(open_params)
        return FsMultiLevelDataset(fs, data_id, **open_params)

    def get_write_data_params_schema(self) -> JsonObjectSchema:
        schema = super().get_write_data_params_schema()
        schema.properties['use_saved_levels'] = JsonBooleanSchema(
            description='Whether to open an already saved level'
                        ' and downscale it then.'
                        ' May be used to avoid computation of'
                        ' entire Dask graphs at each level.'
        )
        return schema

    def write_data(self,
                   data: Union[xr.Dataset, MultiLevelDataset],
                   data_id: str,
                   replace=False,
                   **write_params):
        assert_instance(data, (xr.Dataset, MultiLevelDataset), name='data')
        assert_instance(data_id, str, name='data_id')
        if isinstance(data, MultiLevelDataset):
            ml_dataset = data
        else:
            ml_dataset = BaseMultiLevelDataset(data)
        fs, write_params = self.load_fs(write_params)
        consolidated = write_params.pop('consolidated', True)
        use_saved_levels = write_params.pop('use_saved_levels', False)

        print(f'{data_id}: tile_grid: {ml_dataset.tile_grid}')

        if use_saved_levels:
            ml_dataset = BaseMultiLevelDataset(
                ml_dataset.get_dataset(0),
                tile_grid=ml_dataset.tile_grid
            )

        for index in range(ml_dataset.num_levels):
            level_dataset = ml_dataset.get_dataset(index)

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


class FsMultiLevelDataset(LazyMultiLevelDataset):
    def __init__(self,
                 fs: fsspec.AbstractFileSystem,
                 data_id: str,
                 **open_params: Dict[str, Any]):
        super().__init__(ds_id=data_id)
        self._fs = fs
        self._num_levels = self._get_num_levels(fs, data_id)
        self._open_params = open_params

    def _get_dataset_lazily(self, index: int, parameters) \
            -> xr.Dataset:
        level_path = f'{self.ds_id}/{index}.zarr'
        level_zarr_store = self._fs.get_mapper(level_path)
        consolidated = self._open_params.pop(
            'consolidated',
            self._fs.exists(f'{level_path}/.zmetadata')
        )
        try:
            return xr.open_zarr(level_zarr_store,
                                consolidated=consolidated,
                                **self._open_params)
        except ValueError as e:
            raise DataStoreError(f'Failed to open'
                                 f' dataset {level_path!r}:'
                                 f' {e}') from e

    def _get_tile_grid_lazily(self):
        """
        Retrieve, i.e. read or compute, the tile grid used by the multi-level dataset.

        :return: the dataset for the level at *index*.
        """
        return get_dataset_tile_grid(self.get_dataset(0),
                                     num_levels=self._num_levels)

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
                         for dir_path in fs.listdir(data_id, detail=True)
                         if dir_path['type'] == 'directory'):
            # No ext, i.e. dir_name = "<level>", is proposed by
            # https://github.com/zarr-developers/zarr-specs/issues/50.
            # xcube already selected dir_name = "<level>.zarr".
            basename, ext = os.path.splitext(dir_name)
            if basename and (ext == '' or ext == '.zarr'):
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
