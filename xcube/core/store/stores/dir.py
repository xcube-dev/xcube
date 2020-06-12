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
from xcube.core.store.accessor import find_data_opener_extensions
from xcube.core.store.accessor import find_data_writer_extensions
from xcube.core.store.accessor import get_data_accessor_predicate
from xcube.core.store.accessor import new_data_opener
from xcube.core.store.accessor import new_data_writer
from xcube.core.store.descriptor import DataDescriptor
from xcube.core.store.descriptor import DatasetDescriptor
from xcube.core.store.descriptor import MultiLevelDatasetDescriptor
from xcube.core.store.descriptor import TYPE_ID_DATASET
from xcube.core.store.descriptor import TYPE_ID_GEO_DATA_FRAME
from xcube.core.store.descriptor import TYPE_ID_MULTI_LEVEL_DATASET
from xcube.core.store.descriptor import get_data_type_id
from xcube.core.store.store import DataStoreError
from xcube.core.store.store import MutableDataStore
from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_in
from xcube.util.assertions import assert_instance
from xcube.util.extension import Extension
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema

_EXT_TO_ACCESSOR_ID_PARTS = {
    '.zarr': (TYPE_ID_DATASET, 'zarr', 'posix'),
    '.levels': (TYPE_ID_MULTI_LEVEL_DATASET, 'levels', 'posix'),
    '.nc': (TYPE_ID_DATASET, 'netcdf', 'posix'),
    '.shp': (TYPE_ID_GEO_DATA_FRAME, 'ESRI Shapefile', 'posix'),
    '.geojson': (TYPE_ID_GEO_DATA_FRAME, 'GeoJSON', 'posix'),
}

_TYPE_ID_TO_ACCESSOR_TO_DEFAULT_EXT = {
    TYPE_ID_DATASET: '.zarr',
    TYPE_ID_MULTI_LEVEL_DATASET: '.levels',
    TYPE_ID_GEO_DATA_FRAME: '.geojson'
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
        self._assert_valid_type_id(type_id)
        for data_id in os.listdir(self._base_dir):
            accessor_id_parts = self._get_accessor_id_parts(data_id, require=False)
            if accessor_id_parts:
                actual_type_id, _, _ = accessor_id_parts
                if type_id is None or actual_type_id == type_id:
                    yield data_id

    def get_data_opener_ids(self, type_id: str = None, data_id: str = None) -> Tuple[str, ...]:
        self._assert_valid_type_id(type_id)
        if not type_id and data_id:
            type_id, _, _ = self._get_accessor_id_parts(data_id)
        return tuple(ext.name for ext in find_data_opener_extensions(
            predicate=get_data_accessor_predicate(type_id=type_id, storage_id='posix')
        ))

    def get_open_data_params_schema(self, data_id: str = None, opener_id: str = None) -> JsonObjectSchema:
        if not opener_id and data_id:
            opener_id = self._get_opener_id(data_id)
        return new_data_opener(opener_id).get_open_data_params_schema(data_id=data_id)

    def open_data(self,
                  data_id: str,
                  opener_id: str = None,
                  **open_params) -> xr.Dataset:
        if not opener_id:
            opener_id = self._get_opener_id(data_id)
        path = self._resolve_data_id_to_path(data_id)
        return new_data_opener(opener_id).open_data(path, **open_params)

    def get_data_writer_ids(self, type_id: str = None) -> Tuple[str, ...]:
        self._assert_valid_type_id(type_id)
        extensions = find_data_writer_extensions(
            predicate=get_data_accessor_predicate(type_id=type_id, storage_id='posix')
        )
        return tuple(ext.name for ext in extensions)

    def get_write_data_params_schema(self, writer_id: str = None) -> JsonObjectSchema:
        if not writer_id:
            extensions = find_data_writer_extensions(
                predicate=get_data_accessor_predicate(type_id='dataset', storage_id='posix')
            )
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
                predicate = get_data_accessor_predicate(type_id=TYPE_ID_DATASET,
                                                        format_id='zarr',
                                                        storage_id='posix')
            elif isinstance(data, MultiLevelDataset):
                predicate = get_data_accessor_predicate(type_id=TYPE_ID_MULTI_LEVEL_DATASET,
                                                        format_id='levels',
                                                        storage_id='posix')
            elif isinstance(data, gpd.GeoDataFrame):
                predicate = get_data_accessor_predicate(type_id=TYPE_ID_GEO_DATA_FRAME,
                                                        format_id='geojson',
                                                        storage_id='posix')
            else:
                raise DataStoreError(f'Unsupported data type {type(data)}')
            extensions = find_data_writer_extensions(predicate=predicate)
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

    def _resolve_data_id_to_path(self, data_id: str) -> str:
        assert_given(data_id, 'data_id')
        return os.path.join(self._base_dir, data_id)

    def _assert_valid_type_id(self, type_id: Optional[str]):
        if type_id:
            assert_in(type_id, self.get_type_ids(), 'type_id')

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
        accessor_id_parts = self._get_accessor_id_parts(data_id)
        if not accessor_id_parts:
            if require:
                raise DataStoreError(f'Datasets named "{data_id}" are not supported by this data store')
            return []
        type_id, format_id, storage_id = accessor_id_parts[0]
        extensions = get_data_accessor_extensions(type_id, format_id, storage_id)
        if not extensions:
            if require:
                raise DataStoreError(f'No data accessor found for datasets named "{data_id}"')
            return []
        return extensions

    @classmethod
    def _get_accessor_id_parts(cls, data_id: str, require=True) -> Optional[Tuple[str, str, str]]:
        assert_given(data_id, 'data_id')
        _, ext = os.path.splitext(data_id)
        accessor_id_parts = _EXT_TO_ACCESSOR_ID_PARTS.get(ext)
        if not accessor_id_parts and require:
            raise DataStoreError(f'A dataset named "{data_id}" is not supported')
        return accessor_id_parts

    @classmethod
    def _get_filename_ext(cls, data: Any):
        type_id = get_data_type_id(data)
        return _TYPE_ID_TO_ACCESSOR_TO_DEFAULT_EXT[type_id]
