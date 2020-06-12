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

import uuid
from typing import Iterator, Dict, Any, Optional, Tuple

import xarray as xr

from xcube.core.store.descriptor import DataDescriptor, new_data_descriptor, \
    get_data_type_id
from xcube.core.store.store import MutableDataStore, DataStoreError
from xcube.util.jsonschema import JsonObjectSchema


class MemoryDataStore(MutableDataStore):
    """
    An in-memory cube store.
    Its main use case is testing.
    """

    _GLOBAL_DATA_STORAGE = dict()

    def __init__(self, data_storage: Dict[str, Any] = None):
        self._data_storage = data_storage if data_storage is not None else self.get_global_data_storage()

    #############################################################################
    # MutableDataStore impl.

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema()

    def get_data_ids(self, type_id: str = None) -> Iterator[str]:
        return iter(self._data_storage.keys())

    def describe_data(self, data_id: str) -> DataDescriptor:
        return new_data_descriptor(data_id, self._data_storage[data_id])

    @classmethod
    def get_search_params_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema()

    def search_data(self, type_id: str = None, **search_params) -> Iterator[DataDescriptor]:
        if search_params:
            raise ValueError(f'unsupported open_params {tuple(search_params.keys())}')
        for data_id, data in self._data_storage.items():
            if type_id is None or type_id == get_data_type_id(data):
                yield new_data_descriptor(data_id, data)

    def get_data_opener_ids(self, type_id: str = None, data_id: str = None) -> Tuple[str, ...]:
        return ()

    def get_open_data_params_schema(self, data_id: str = None, opener_id: str = None) -> JsonObjectSchema:
        return JsonObjectSchema()

    def open_data(self, data_id: str, opener_id: str = None, **open_params) -> Any:
        if open_params:
            raise ValueError(f'unsupported open_params {tuple(open_params.keys())}')
        if data_id not in self._data_storage:
            raise DataStoreError(f'data resource "{data_id}" does not exist in store')
        return self._data_storage[data_id]

    def get_data_writer_ids(self, type_id: str = None) -> Tuple[str, ...]:
        return ()

    def get_write_data_params_schema(self, writer_id: str = None) -> JsonObjectSchema:
        return JsonObjectSchema()

    def write_data(self, data: Any, data_id: str = None, writer_id: str = None, replace: bool = False,
                   **write_params) -> str:
        if write_params:
            raise ValueError(f'unsupported write_params {tuple(write_params.keys())}')
        data_id = self._ensure_valid_data_id(data_id)
        if data_id in self._data_storage and not replace:
            raise DataStoreError(f'data resource "{data_id}" already exist in store')
        self._data_storage[data_id] = data
        return data_id

    def delete_data(self, data_id: str):
        if data_id not in self._data_storage:
            raise DataStoreError(f'data resource "{data_id}" does not exist in store')
        del self._data_storage[data_id]

    def register_data(self, data_id: str, data: Any):
        # Not required
        pass

    def deregister_data(self, data_id: str):
        # Not required
        pass

    #############################################################################
    # Specific interface

    @property
    def data_storage(self) -> Dict[str, xr.Dataset]:
        return self._data_storage

    @classmethod
    def get_global_data_storage(cls) -> Dict[str, xr.Dataset]:
        return cls._GLOBAL_DATA_STORAGE

    @classmethod
    def replace_global_cube_memory(cls, global_cube_memory: Dict[str, xr.Dataset]) -> Dict[str, xr.Dataset]:
        old_global_cube_memory = cls._GLOBAL_DATA_STORAGE
        cls._GLOBAL_DATA_STORAGE = global_cube_memory
        return old_global_cube_memory

    @classmethod
    def _ensure_valid_data_id(cls, data_id: Optional[str]) -> str:
        return data_id or str(uuid.uuid4())
