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
from typing import Iterator, Dict, Any, Optional, Tuple, Mapping

from xcube.core.store.descriptor import DataDescriptor
from xcube.core.store.descriptor import get_data_type_id
from xcube.core.store.descriptor import new_data_descriptor
from xcube.core.store.store import MutableDataStore, DataStoreError
from xcube.util.assertions import assert_given
from xcube.util.jsonschema import JsonObjectSchema


# TODO: complete docs

class MemoryDataStore(MutableDataStore):
    """
    An in-memory cube store.
    Its main use case is testing.
    """

    _GLOBAL_DATA_DICT = dict()

    def __init__(self, data_dict: Dict[str, Any] = None):
        self._data_dict = data_dict if data_dict is not None else self.get_global_data_dict()

    #############################################################################
    # MutableDataStore impl.

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema()

    @classmethod
    def get_type_ids(cls) -> Tuple[str, ...]:
        return '*',

    def get_data_ids(self, type_id: str = None) -> Iterator[str]:
        return iter(self._data_dict.keys())

    def has_data(self, data_id: str) -> bool:
        assert_given(data_id, 'data_id')
        return data_id in self._data_dict

    def describe_data(self, data_id: str) -> DataDescriptor:
        self._assert_valid_data_id(data_id)
        return new_data_descriptor(data_id, self._data_dict[data_id])

    @classmethod
    def get_search_params_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema()

    def search_data(self, type_id: str = None, **search_params) -> Iterator[DataDescriptor]:
        self._assert_empty_params(search_params, 'search_params')
        for data_id, data in self._data_dict.items():
            if type_id is None or type_id == get_data_type_id(data):
                yield new_data_descriptor(data_id, data)

    def get_data_opener_ids(self, data_id: str = None, type_id: str = None) -> Tuple[str, ...]:
        return '*:*:memory',

    def get_open_data_params_schema(self, data_id: str = None, opener_id: str = None) -> JsonObjectSchema:
        return JsonObjectSchema()

    def open_data(self, data_id: str, opener_id: str = None, **open_params) -> Any:
        self._assert_valid_data_id(data_id)
        self._assert_empty_params(open_params, 'open_params')
        return self._data_dict[data_id]

    def get_data_writer_ids(self, type_id: str = None) -> Tuple[str, ...]:
        return '*:*:memory',

    def get_write_data_params_schema(self, writer_id: str = None) -> JsonObjectSchema:
        return JsonObjectSchema()

    def write_data(self, data: Any, data_id: str = None, writer_id: str = None, replace: bool = False,
                   **write_params) -> str:
        self._assert_empty_params(write_params, 'write_params')
        data_id = self._ensure_valid_data_id(data_id)
        if data_id in self._data_dict and not replace:
            raise DataStoreError(f'Data resource "{data_id}" already exist in store')
        self._data_dict[data_id] = data
        return data_id

    def delete_data(self, data_id: str):
        self._assert_valid_data_id(data_id)
        del self._data_dict[data_id]

    def register_data(self, data_id: str, data: Any):
        # Not required
        pass

    def deregister_data(self, data_id: str):
        # Not required
        pass

    #############################################################################
    # Specific interface

    @property
    def data_dict(self) -> Dict[str, Any]:
        return self._data_dict

    @classmethod
    def get_global_data_dict(cls) -> Dict[str, Any]:
        return cls._GLOBAL_DATA_DICT

    @classmethod
    def replace_global_data_dict(cls, global_cube_memory: Dict[str, Any]) -> Dict[str, Any]:
        old_global_cube_memory = cls._GLOBAL_DATA_DICT
        cls._GLOBAL_DATA_DICT = global_cube_memory
        return old_global_cube_memory

    #############################################################################
    # Implementation helpers

    @classmethod
    def _ensure_valid_data_id(cls, data_id: Optional[str]) -> str:
        return data_id or str(uuid.uuid4())

    def _assert_valid_data_id(self, data_id):
        assert_given(data_id, 'data_id')
        if data_id not in self._data_dict:
            raise DataStoreError(f'Data resource "{data_id}" does not exist in store')

    def _assert_empty_params(self, params: Optional[Mapping[str, Any]], name: str):
        if params:
            param_names = ', '.join(map(lambda k: f'"{k}"', params.keys()))
            raise DataStoreError(f'Unsupported {name} {param_names}')
