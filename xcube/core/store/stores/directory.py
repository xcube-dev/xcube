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
from typing import Optional, Iterator, Any, Tuple, List

import geopandas as gpd
import xarray as xr

from xcube.core.mldataset import MultiLevelDataset
from xcube.core.store import DataDescriptor
from xcube.core.store import DataStoreError
from xcube.core.store import MutableDataStore
from xcube.core.store import TYPE_SPECIFIER_ANY
from xcube.core.store import TYPE_SPECIFIER_DATASET
from xcube.core.store import TYPE_SPECIFIER_GEODATAFRAME
from xcube.core.store import TYPE_SPECIFIER_MULTILEVEL_DATASET
from xcube.core.store import TypeSpecifier
from xcube.core.store import find_data_opener_extensions
from xcube.core.store import DefaultSearchMixin
from xcube.core.store import find_data_writer_extensions
from xcube.core.store import get_data_accessor_predicate
from xcube.core.store import get_type_specifier
from xcube.core.store import new_data_descriptor
from xcube.core.store import new_data_opener
from xcube.core.store import new_data_writer
from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_in
from xcube.util.assertions import assert_instance
from xcube.util.extension import Extension
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema

_STORAGE_ID = 'posix'

_DEFAULT_FORMAT_ID = 'zarr'

_FILENAME_EXT_TO_ACCESSOR_ID_PARTS = {
    '.zarr': (str(TYPE_SPECIFIER_DATASET), 'zarr', _STORAGE_ID),
    '.levels': (str(TYPE_SPECIFIER_MULTILEVEL_DATASET), 'levels', _STORAGE_ID),
    '.nc': (str(TYPE_SPECIFIER_DATASET), 'netcdf', _STORAGE_ID),
    '.shp': (str(TYPE_SPECIFIER_GEODATAFRAME), 'shapefile', _STORAGE_ID),
    '.geojson': (str(TYPE_SPECIFIER_GEODATAFRAME), 'geojson', _STORAGE_ID),
}

_TYPE_SPECIFIER_TO_ACCESSOR_TO_DEFAULT_FILENAME_EXT = {
    TYPE_SPECIFIER_DATASET: '.zarr',
    TYPE_SPECIFIER_MULTILEVEL_DATASET: '.levels',
    TYPE_SPECIFIER_GEODATAFRAME: '.geojson'
}


# TODO: write tests
# TODO: complete docs
# TODO: remove code duplication with ./s3.py and its tests.
#   - Introduce a file-system-abstracting base class or mixin, see module "fsspec" and impl. "s3fs" as  used in Dask!
#   - Introduce something like MultiOpenerStoreMixin/MultiWriterStoreMixin!

class DirectoryDataStore(DefaultSearchMixin, MutableDataStore):
    """
    A cube store that stores cubes in a directory in the local file system.

    :param base_dir: The base directory where cubes are stored.
    :param read_only: Whether this is a read-only store.
    """

    def __init__(self,
                 base_dir: str = None,
                 read_only: bool = False):
        assert_given(base_dir, 'base_dir')
        self._base_dir = base_dir
        self._read_only = read_only

    @property
    def base_dir(self) -> Optional[str]:
        return self._base_dir

    @property
    def read_only(self) -> bool:
        return self._read_only

    #############################################################################
    # MutableDataStore impl.

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                base_dir=JsonStringSchema(min_length=1),
                read_only=JsonBooleanSchema(default=False)
            ),
            required=['base_dir'],
            additional_properties=False
        )

    @classmethod
    def get_type_specifiers(cls) -> Tuple[str, ...]:
        return str(TYPE_SPECIFIER_DATASET), str(TYPE_SPECIFIER_MULTILEVEL_DATASET), str(TYPE_SPECIFIER_GEODATAFRAME)

    def get_type_specifiers_for_data(self, data_id: str) -> Tuple[str, ...]:
        self._assert_valid_data_id(data_id)
        actual_type_specifier, _, _ = self._get_accessor_id_parts(data_id)
        return actual_type_specifier,

    def get_data_ids(self, type_specifier: str = None, include_titles: bool = True) -> \
            Iterator[Tuple[str, Optional[str]]]:
        if type_specifier is not None:
            type_specifier = TypeSpecifier.normalize(type_specifier)
        # TODO: Use os.walk(), which provides a generator rather than a list
        # os.listdir does not guarantee any ordering of the entries, so
        # sort them to ensure predictable behaviour.
        for data_id in sorted(os.listdir(self._base_dir)):
            actual_type_specifier = self._get_type_specifier_for_data_id(data_id, require=False)
            if actual_type_specifier is not None:
                if type_specifier is None or actual_type_specifier.satisfies(type_specifier):
                    yield data_id, None

    def has_data(self, data_id: str, type_specifier: str = None) -> bool:
        assert_given(data_id, 'data_id')
        actual_type_specifier = self._get_type_specifier_for_data_id(data_id)
        if actual_type_specifier is not None:
            if type_specifier is None or actual_type_specifier.satisfies(type_specifier):
                path = self._resolve_data_id_to_path(data_id)
                return os.path.exists(path)
        return False

    def describe_data(self, data_id: str, type_specifier: str = None) -> DataDescriptor:
        self._assert_valid_data_id(data_id)
        actual_type_specifier = self._get_type_specifier_for_data_id(data_id)
        if actual_type_specifier is not None:
            if type_specifier is None or actual_type_specifier.satisfies(type_specifier):
                data = self.open_data(data_id)
                return new_data_descriptor(data_id, data, require=True)
            else:
                raise DataStoreError(f'Type specifier "{type_specifier}" cannot be satisfied'
                                     f' by type specifier "{actual_type_specifier}" of data resource "{data_id}"')
        else:
            raise DataStoreError(f'Data resource "{data_id}" not found')

    def get_data_opener_ids(self, data_id: str = None, type_specifier: Optional[str] = None) -> Tuple[str, ...]:
        if type_specifier:
            type_specifier = TypeSpecifier.parse(type_specifier)
        if type_specifier == TYPE_SPECIFIER_ANY:
            type_specifier = None
        self._assert_valid_type_specifier(type_specifier)
        if not type_specifier and data_id:
            type_specifier, _, _ = self._get_accessor_id_parts(data_id)
        return tuple(ext.name for ext in find_data_opener_extensions(
            predicate=get_data_accessor_predicate(type_specifier=type_specifier, storage_id=_STORAGE_ID)
        ))

    def get_open_data_params_schema(self, data_id: str = None, opener_id: str = None) -> JsonObjectSchema:
        if not opener_id and data_id:
            opener_id = self._get_opener_id(data_id)
        if not opener_id:
            extensions = find_data_opener_extensions(
                predicate=get_data_accessor_predicate(type_specifier='dataset',
                                                      format_id=_DEFAULT_FORMAT_ID,
                                                      storage_id=_STORAGE_ID)
            )
            assert extensions
            opener_id = extensions[0].name
        return new_data_opener(opener_id).get_open_data_params_schema(data_id=data_id)

    def open_data(self,
                  data_id: str,
                  opener_id: str = None,
                  **open_params) -> xr.Dataset:
        self._assert_valid_data_id(data_id)
        if not opener_id:
            opener_id = self._get_opener_id(data_id)
        path = self._resolve_data_id_to_path(data_id)
        return new_data_opener(opener_id).open_data(path, **open_params)

    def get_data_writer_ids(self, type_specifier: str = None) -> Tuple[str, ...]:
        if type_specifier:
            type_specifier = TypeSpecifier.parse(type_specifier)
        if type_specifier == TYPE_SPECIFIER_ANY:
            type_specifier = None
        self._assert_valid_type_specifier(type_specifier)
        extensions = find_data_writer_extensions(
            predicate=get_data_accessor_predicate(type_specifier=type_specifier, storage_id=_STORAGE_ID)
        )
        return tuple(ext.name for ext in extensions)

    def get_write_data_params_schema(self, writer_id: str = None) -> JsonObjectSchema:
        if not writer_id:
            extensions = find_data_writer_extensions(
                predicate=get_data_accessor_predicate(type_specifier='dataset',
                                                      format_id=_DEFAULT_FORMAT_ID,
                                                      storage_id=_STORAGE_ID)
            )
            assert extensions
            writer_id = extensions[0].name
        return new_data_writer(writer_id).get_write_data_params_schema()

    def write_data(self,
                   data: Any,
                   data_id: str = None,
                   writer_id: str = None,
                   replace: bool = False,
                   **write_params) -> str:
        assert_instance(data, (xr.Dataset, MultiLevelDataset, gpd.GeoDataFrame))
        if not writer_id:
            if isinstance(data, xr.Dataset):
                predicate = get_data_accessor_predicate(type_specifier=TYPE_SPECIFIER_DATASET,
                                                        format_id='zarr',
                                                        storage_id=_STORAGE_ID)
            elif isinstance(data, MultiLevelDataset):
                predicate = get_data_accessor_predicate(type_specifier=TYPE_SPECIFIER_MULTILEVEL_DATASET,
                                                        format_id='levels',
                                                        storage_id=_STORAGE_ID)
            elif isinstance(data, gpd.GeoDataFrame):
                predicate = get_data_accessor_predicate(type_specifier=TYPE_SPECIFIER_GEODATAFRAME,
                                                        format_id='geojson',
                                                        storage_id=_STORAGE_ID)
            else:
                raise DataStoreError(f'Unsupported data type "{type(data)}"')
            extensions = find_data_writer_extensions(predicate=predicate)
            assert extensions
            writer_id = extensions[0].name
        data_id = self._ensure_valid_data_id(data_id, data)
        path = self._resolve_data_id_to_path(data_id)
        new_data_writer(writer_id).write_data(data, path, replace=replace, **write_params)
        return data_id

    def delete_data(self, data_id: str):
        accessor_id_parts = self._get_accessor_id_parts(data_id)
        writer_id = ':'.join(accessor_id_parts)
        path = self._resolve_data_id_to_path(data_id)
        new_data_writer(writer_id).delete_data(path)

    def register_data(self, data_id: str, data: Any):
        # We don't need this as we use the file system
        pass

    def deregister_data(self, data_id: str):
        # We don't need this as we use the file system
        pass

    ###############################################################
    # Implementation helpers

    @classmethod
    def _ensure_valid_data_id(cls, data_id: Optional[str], data: Any) -> str:
        return data_id or str(uuid.uuid4()) + cls._get_filename_ext(data)

    def _assert_valid_data_id(self, data_id):
        if not self.has_data(data_id):
            raise DataStoreError(f'Data resource "{data_id}" does not exist in store')

    def _resolve_data_id_to_path(self, data_id: str) -> str:
        assert_given(data_id, 'data_id')
        return os.path.join(self._base_dir, data_id)

    def _assert_valid_type_specifier(self, type_specifier: Optional[TypeSpecifier]):
        if type_specifier is not None:
            assert_in(type_specifier, self.get_type_specifiers(), 'type_specifier')

    def _get_opener_id(self, data_id: str):
        return self._get_accessor_id(data_id, find_data_opener_extensions)

    def _get_writer_id(self, data_id: str):
        return self._get_accessor_id(data_id, find_data_writer_extensions)

    def _get_opener_extensions(self, data_id: str, require=True):
        return self._get_accessor_extensions(data_id, find_data_opener_extensions, require=require)

    def _get_writer_extensions(self, data_id: str, require=True):
        return self._get_accessor_extensions(data_id, find_data_writer_extensions, require=require)

    def _get_accessor_id(self, data_id: str, get_data_accessor_extensions, require=True) -> Optional[str]:
        extensions = self._get_accessor_extensions(data_id, get_data_accessor_extensions, require=require)
        return extensions[0].name if extensions else None

    def _get_accessor_extensions(self, data_id: str, get_data_accessor_extensions, require=True) -> List[Extension]:
        accessor_id_parts = self._get_accessor_id_parts(data_id, require=require)
        if not accessor_id_parts:
            return []
        type_specifier, format_id, storage_id = accessor_id_parts
        predicate = get_data_accessor_predicate(type_specifier=type_specifier,
                                                format_id=format_id,
                                                storage_id=storage_id)
        extensions = get_data_accessor_extensions(predicate)
        if not extensions:
            if require:
                raise DataStoreError(f'No accessor found for data resource "{data_id}"')
            return []
        return extensions

    def _get_type_specifier_for_data_id(self, data_id: str, require=True) -> Optional[TypeSpecifier]:
        accessor_id_parts = self._get_accessor_id_parts(data_id, require=require)
        if accessor_id_parts is None:
            return None
        actual_type_specifier, _, _ = accessor_id_parts
        return TypeSpecifier.parse(actual_type_specifier)

    @classmethod
    def _get_accessor_id_parts(cls, data_id: str, require=True) -> Optional[Tuple[str, str, str]]:
        assert_given(data_id, 'data_id')
        _, ext = os.path.splitext(data_id)
        accessor_id_parts = _FILENAME_EXT_TO_ACCESSOR_ID_PARTS.get(ext)
        if not accessor_id_parts and require:
            raise DataStoreError(f'A dataset named "{data_id}" is not supported')
        return accessor_id_parts

    @classmethod
    def _get_filename_ext(cls, data: Any):
        type_specifier = get_type_specifier(data)
        return _TYPE_SPECIFIER_TO_ACCESSOR_TO_DEFAULT_FILENAME_EXT[type_specifier]
