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

from xcube.core.store import DataDescriptor
from xcube.core.store import DataStoreError
from xcube.core.store import get_type_specifier
from xcube.core.store import MutableDataStore
from xcube.core.store import TypeSpecifier
from xcube.core.store import TYPE_SPECIFIER_ANY
from xcube.core.store import new_data_descriptor
from xcube.util.assertions import assert_given
from xcube.util.jsonschema import JsonObjectSchema

_STORAGE_ID = 'memory'
_ACCESSOR_ID = f'*:*:{_STORAGE_ID}'


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
    def get_type_specifiers(cls) -> Tuple[str, ...]:
        return str(TYPE_SPECIFIER_ANY),

    def get_type_specifiers_for_data(self, data_id: str) -> Tuple[str, ...]:
        self._assert_valid_data_id(data_id)
        type_specifier = get_type_specifier(self._data_dict[data_id])
        return str(type_specifier),

    def get_data_ids(self, type_specifier: str = None, include_titles: bool = True) -> Iterator[Tuple[str, Optional[str]]]:
        if type_specifier is None:
            for data_id, data in self._data_dict.items():
                yield data_id, None
        else:
            type_specifier = TypeSpecifier.normalize(type_specifier)
            for data_id, data in self._data_dict.items():
                data_type_specifier = get_type_specifier(data)
                if data_type_specifier is None or data_type_specifier.satisfies(type_specifier):
                    yield data_id, None

    def has_data(self, data_id: str, type_specifier: str = None) -> bool:
        assert_given(data_id, 'data_id')
        if data_id not in self._data_dict:
            return False
        if type_specifier is not None:
            data_type_specifier = get_type_specifier(self._data_dict[data_id])
            if data_type_specifier is None or not data_type_specifier.satisfies(type_specifier):
                return False
        return True

    def describe_data(self, data_id: str, type_specifier: str = None) -> DataDescriptor:
        self._assert_valid_data_id(data_id)
        if type_specifier is not None:
            data_type_specifier = get_type_specifier(self._data_dict[data_id])
            if data_type_specifier is None or not data_type_specifier.satisfies(type_specifier):
                raise DataStoreError(f'Type specifier "{type_specifier}" cannot be satisfied'
                                     f' by type specifier "{data_type_specifier}" of data resource "{data_id}"')
        return new_data_descriptor(data_id, self._data_dict[data_id])

    @classmethod
    def get_search_params_schema(self, type_specifier: str = None) -> JsonObjectSchema:
        return JsonObjectSchema()

    def search_data(self, type_specifier: str = None, **search_params) -> Iterator[DataDescriptor]:
        self._assert_empty_params(search_params, 'search_params')
        for data_id, _ in self.get_data_ids(type_specifier=type_specifier):
            yield new_data_descriptor(data_id, self._data_dict[data_id])

    def get_data_opener_ids(self, data_id: str = None, type_specifier: str = None) -> Tuple[str, ...]:
        return _ACCESSOR_ID,

    def get_open_data_params_schema(self, data_id: str = None, opener_id: str = None) -> JsonObjectSchema:
        return JsonObjectSchema()

    def open_data(self, data_id: str, opener_id: str = None, **open_params) -> Any:
        self._assert_valid_data_id(data_id)
        self._assert_empty_params(open_params, 'open_params')
        return self._data_dict[data_id]

    def get_data_writer_ids(self, type_specifier: str = None) -> Tuple[str, ...]:
        return _ACCESSOR_ID,

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
