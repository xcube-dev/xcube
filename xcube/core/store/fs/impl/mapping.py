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

import collections.abc
import collections.abc
from typing import Union, Tuple, Dict, Any

import fsspec
import xarray as xr

from xcube.core.mldataset import MultiLevelDataset
from xcube.util.assertions import assert_instance
from .dataset import DatasetGeoTiffFsDataAccessor
from .dataset import DatasetNetcdfFsDataAccessor
from .dataset import DatasetZarrFsDataAccessor
from .mldataset import FsMultiLevelBase
from .mldataset import MultiLevelDatasetLevelsFsDataAccessor
from ...datatype import DataType
from ...datatype import MAPPING_TYPE


class MappingZarrFsDataAccessor(DatasetZarrFsDataAccessor):
    """
    Opener extension name: "mapping:zarr:<protocol>".
    """

    force_mapping = True


class MappingNetcdfFsDataAccessor(DatasetNetcdfFsDataAccessor):
    """
    Opener extension name: "mapping:netcdf:<protocol>".
    """

    force_mapping = True


class MappingGeoTiffFsDataAccessor(DatasetGeoTiffFsDataAccessor):
    """
    Opener extension name: "mapping:geotiff:<protocol>".
    """

    force_mapping = True


class MappingLevelsFsDataAccessor(
    MultiLevelDatasetLevelsFsDataAccessor
):
    """
    Opener extension name: "mapping:levels:<protocol>".
    """

    @classmethod
    def get_data_types(cls) -> Tuple[DataType, ...]:
        return MAPPING_TYPE,

    def open_data(self, path: str, **open_params) \
            -> collections.abc.Mapping:
        assert_instance(path, str, name='path')
        fs, root, open_params = self.load_fs(open_params)
        return FsMultiLevelZarrStore(fs, path, **open_params)

    def write_data(self,
                   data: Union[xr.Dataset, MultiLevelDataset],
                   data_id: str,
                   replace: bool = False,
                   **write_params) -> str:
        assert_instance(data, collections.abc.Mapping, name='data')
        assert_instance(data_id, str, name='data_id')
        raise NotImplementedError()


class FsMultiLevelZarrStore(collections.abc.MutableMapping[str, bytes]):
    def __init__(self,
                 fs: fsspec.AbstractFileSystem,
                 path: str,
                 **open_params: Dict[str, Any]):
        self._base = FsMultiLevelBase(fs,
                                      path,
                                      lock=None,
                                      **open_params)

    def __len__(self) -> int:
        length = 0
        for level in range(self._base.num_levels):
            zarr_store, _ = self._base.get_zarr_store(level)
            length += 1 + len(zarr_store)
        return length

    def __iter__(self):
        for level in range(self._base.num_levels):
            yield f"{level}"
            zarr_store, _ = self._base.get_zarr_store(level)
            for key in iter(zarr_store):
                yield f"{level}/{key}"

    def __getitem__(self, key: str):
        level, store_key = self._parse_key(key)
        if store_key is None:
            # This is the level "directory", where int(key) == level
            return b""
        zarr_store, _ = self._base.get_zarr_store(level)
        return zarr_store[store_key]

    def __setitem__(self, key: str, value: bytes):
        raise NotImplementedError()

    def __delitem__(self, key: str):
        raise NotImplementedError()

    def __contains__(self, key: str):
        try:
            level, store_key = self._parse_key(key)
        except KeyError:
            return False
        if store_key is None:
            return True
        zarr_store, _ = self._base.get_zarr_store(level)
        return store_key in zarr_store

    def _parse_key(self, key: str) -> Tuple[int, str]:
        """Turn key into level index and store key."""
        try:
            index, store_key = int(key), None
        except ValueError:
            try:
                index, store_key = key.split('/', maxsplit=1)
                index = int(index)
            except ValueError:
                raise KeyError(key)
        if not (0 <= index < self._base.num_levels):
            raise KeyError(key)
        return index, store_key
