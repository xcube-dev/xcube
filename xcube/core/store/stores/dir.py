# The MIT License (MIT)
# Copyright (c) 2020 by the xcube development team and contributors
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
import uuid
from typing import Optional, Iterator, Any, Tuple

import xarray as xr

from xcube.core.mldataset import MultiLevelDataset
from xcube.core.store.accessor_v4 import get_data_opener
from xcube.core.store.accessor_v4 import get_data_opener_infos
from xcube.core.store.accessor_v4 import get_data_writer
from xcube.core.store.accessor_v4 import get_data_writer_infos
from xcube.core.store.descriptor import DataDescriptor
from xcube.core.store.descriptor import DatasetDescriptor
from xcube.core.store.descriptor import MultiLevelDatasetDescriptor
from xcube.core.store.store_v4 import DataStoreError
from xcube.core.store.store_v4 import MutableDataStore
from xcube.util.assertions import assert_in, assert_instance
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema

_EXT_TO_ACCESSOR_ID_PARTS = {
    '.zarr': ('dataset', 'zarr', 'posix'),
    '.levels': ('mldataset', 'levels', 'posix'),
    '.nc': ('dataset', 'netcdf', 'posix'),
}


# TODO: validate params
# TODO: complete tests
class DirectoryDataStore(MutableDataStore):
    """
    A cube store that stores cubes in a directory in the local file system.

    :param base_dir: The base directory where cubes are stored.
    :param read_only: Whether this is a read-only store.
    """

    def __init__(self,
                 base_dir: str,
                 read_only: bool = False):
        self._base_dir = base_dir
        self._read_only = read_only

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                base_dir=JsonStringSchema(default='.'),
                read_only=JsonBooleanSchema(default=False)
            ),
            additional_properties=False
        )

    @classmethod
    def get_type_ids(cls) -> Tuple[str, ...]:
        return 'dataset', 'mldataset'

    def describe_data(self, data_id: str) -> DataDescriptor:
        if data_id.endswith('.levels'):
            # TODO: implement me
            return DatasetDescriptor(data_id=data_id)
        else:
            # TODO: implement me
            return MultiLevelDatasetDescriptor(data_id=data_id, num_levels=6)

    @classmethod
    def get_search_params_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema()

    def search_data(self, type_id: str = None, **search_params) -> Iterator[DataDescriptor]:
        pass

    def get_data_ids(self, type_id: str = None) -> Iterator[str]:
        self._assert_type_id(type_id)
        for data_id in os.listdir(self._base_dir):
            accessor_id_parts = self._get_accessor_id_parts(data_id)
            if accessor_id_parts:
                actual_type_id, _, _ = accessor_id_parts
                if type_id is None or actual_type_id == type_id:
                    yield data_id

    @classmethod
    def _get_accessor_id_parts(cls, data_id) -> Optional[Tuple[str, str, str]]:
        _, ext = os.path.splitext(data_id)
        return _EXT_TO_ACCESSOR_ID_PARTS.get(ext)

    def get_data_opener_ids(self, type_id: str = None, data_id: str = None) -> Tuple[str, ...]:
        self._assert_type_id(type_id)
        format_id = None
        storage_id = 'posix'
        if not type_id and data_id:
            accessor_id_parts = self._get_accessor_id_parts(data_id)
            if accessor_id_parts:
                type_id, format_id, storage_id = accessor_id_parts[0]
        return tuple(get_data_opener_infos(type_id=type_id, format_id=format_id, storage_id=storage_id).keys())

    def get_open_data_params_schema(self, data_id: str = None, opener_id: str = None) -> JsonObjectSchema:
        return JsonObjectSchema(properties=dict(
            decode_cf=JsonBooleanSchema(default=True),
            format=JsonStringSchema(nullable=True, default=None),
        ))

    def open_data(self,
                  data_id: str,
                  opener_id: str = None,
                  **open_params) -> xr.Dataset:
        if not opener_id:
            accessor_id_parts = self._get_accessor_id_parts(data_id)
            if not accessor_id_parts:
                raise DataStoreError(f'A dataset named "{data_id}" is not supported')
            type_id, format_id, storage_id = accessor_id_parts[0]
            opener_ids = tuple(get_data_opener_infos(type_id, format_id, storage_id).keys())
            opener_id = opener_ids[0]
        path = self.resolve_data_id_to_path(data_id)
        return get_data_opener(opener_id).open_data(path, **open_params)

    def get_data_writer_ids(self, type_id: str = None) -> Tuple[str, ...]:
        self._assert_type_id(type_id)
        return tuple(get_data_writer_infos(type_id=type_id, format_id='posix').keys())

    def get_write_data_params_schema(self, writer_id: str = None) -> JsonObjectSchema:
        return JsonObjectSchema(properties=dict(
            format=JsonStringSchema(default='zarr'),
        ))

    def write_data(self,
                   data: Any,
                   data_id: str = None,
                   writer_id: str = None,
                   replace: bool = False,
                   **write_params) -> str:
        assert_instance(data, (xr.Dataset, MultiLevelDataset))
        if not writer_id:
            if isinstance(data, xr.Dataset):
                type_id = 'dataset'
            else:
                type_id = 'mldataset'
            writer_ids = tuple(get_data_writer_infos(type_id, 'zarr', 'posix').keys())
            writer_id = writer_ids[0]
        data_id = self._ensure_valid_data_id(data_id)
        path = self.resolve_data_id_to_path(data_id)
        get_data_writer(writer_id).write_data(data, path, replace=replace, **write_params)
        return data_id

    def delete_data(self, data_id: str):
        accessor_id_parts = self._get_accessor_id_parts(data_id)
        if not accessor_id_parts:
            raise DataStoreError(f'A dataset named "{data_id}" is not supported')
        writer_id = ':'.join(accessor_id_parts)
        path = self.resolve_data_id_to_path(data_id)
        get_data_writer(writer_id).delete_data(path)

    def register_data(self, data_id: str, data: Any):
        pass

    def deregister_data(self, data_id: str):
        pass

    @classmethod
    def _ensure_valid_data_id(cls, cube_id: Optional[str]) -> str:
        return cube_id or str(uuid.uuid4())

    @property
    def base_dir(self) -> str:
        return self._base_dir

    @property
    def read_only(self) -> bool:
        return self._read_only

    def resolve_data_id_to_path(self, cube_id: str) -> str:
        if not cube_id:
            raise DataStoreError(f'Missing cube identifier')
        return os.path.join(self._base_dir, cube_id)

    def _assert_type_id(self, type_id: Optional[str]):
        if type_id:
            assert_in(type_id, self.get_type_ids(), 'type_id')
