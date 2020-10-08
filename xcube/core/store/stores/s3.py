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
from xcube.core.store import TYPE_ID_ANY
from xcube.core.store import TYPE_ID_DATASET
from xcube.core.store import TYPE_ID_MULTI_LEVEL_DATASET
from xcube.core.store import find_data_opener_extensions
from xcube.core.store import find_data_writer_extensions
from xcube.core.store import get_data_accessor_predicate
from xcube.core.store import get_type_id
from xcube.core.store import new_data_opener
from xcube.core.store import new_data_writer
from xcube.core.store import TypeId
from xcube.core.store.accessors.dataset import S3Mixin
from xcube.util.assertions import assert_condition
from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_in
from xcube.util.assertions import assert_instance
from xcube.util.extension import Extension
from xcube.util.jsonschema import JsonObjectSchema

_STORAGE_ID = 's3'

_DEFAULT_FORMAT_ID = 'zarr'

_FILENAME_EXT_TO_ACCESSOR_ID_PARTS = {
    '.zarr': (TYPE_ID_DATASET, 'zarr', _STORAGE_ID),
    '.levels': (TYPE_ID_MULTI_LEVEL_DATASET, 'levels', _STORAGE_ID),
}

_TYPE_ID_TO_ACCESSOR_TO_DEFAULT_FILENAME_EXT = {
    TYPE_ID_DATASET: '.zarr',
    TYPE_ID_MULTI_LEVEL_DATASET: '.levels',
}


# TODO: write tests
# TODO: complete docs
# TODO: implement '*.levels' support
# TODO: remove code duplication with ./directory.py and its tests.
#   - Introduce a file-system-abstracting base class or mixin, see module "fsspec" and impl. "s3fs" as  used in Dask!
#   - Introduce something like MultiOpenerStoreMixin/MultiWriterStoreMixin!

class S3DataStore(MutableDataStore):
    """
    A cube store that stores cubes in a directory in the local file system.

    :param anon: Anonymous access.
    :param aws_access_key_id: Optional AWS access key identifier.
    :param aws_secret_access_key: Optional AWS secret access key.
    :param aws_session_token: Optional AWS session token.
    :param bucket_name: Mandatory bucket name.
    :param region_name: Optional region name.
    """

    def __init__(self, **store_params):
        self._s3_fs, store_params = S3Mixin.consume_s3fs_params(store_params)
        self._bucket_name, store_params = S3Mixin.consume_bucket_name_param(store_params)
        assert_given(self._bucket_name, 'bucket_name')
        assert_condition(not store_params, f'Unknown keyword arguments: {", ".join(store_params.keys())}')

    def close(self):
        self._s3_fs = None

    #############################################################################
    # MutableDataStore impl.

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        schema = S3Mixin.get_s3_params_schema()
        schema.required.add('bucket_name')
        return schema

    @classmethod
    def get_type_ids(cls) -> Tuple[str, ...]:
        return str(TYPE_ID_DATASET),

    def get_data_ids(self, type_id: str = None) -> Iterator[Tuple[str, Optional[str]]]:
        # todo do not ignore type_id
        prefix = self._bucket_name + '/'
        first_index = len(prefix)
        for item in self._s3_fs.listdir(self._bucket_name, detail=False):
            if item.startswith(prefix):
                yield item[first_index:], None

    def has_data(self, data_id: str) -> bool:
        path = self._resolve_data_id_to_path(data_id)
        return self._s3_fs.exists(path)

    def describe_data(self, data_id: str) -> DataDescriptor:
        # TODO: implement me
        raise NotImplementedError()

    @classmethod
    def get_search_params_schema(cls) -> JsonObjectSchema:
        # TODO: implement me
        raise NotImplementedError()

    def search_data(self, type_id: str = None, **search_params) -> Iterator[DataDescriptor]:
        # TODO: implement me
        raise NotImplementedError()

    def get_data_opener_ids(self, data_id: str = None, type_id: str = None) -> Tuple[str, ...]:
        if type_id:
            type_id = TypeId.normalize(type_id)
        if type_id == TYPE_ID_ANY:
            type_id = None
        self._assert_valid_type_id(type_id)
        if not type_id and data_id:
            type_id, _, _ = self._get_accessor_id_parts(data_id)
        return tuple(ext.name for ext in find_data_opener_extensions(
            predicate=get_data_accessor_predicate(type_id=type_id, storage_id=_STORAGE_ID)
        ))

    def get_open_data_params_schema(self, data_id: str = None, opener_id: str = None) -> JsonObjectSchema:
        if not opener_id and data_id:
            opener_id = self._get_opener_id(data_id)
        if not opener_id:
            extensions = find_data_opener_extensions(
                predicate=get_data_accessor_predicate(type_id='dataset',
                                                      format_id=_DEFAULT_FORMAT_ID,
                                                      storage_id=_STORAGE_ID)
            )
            assert extensions
            opener_id = extensions[0].name
        return self._new_s3_opener(opener_id).get_open_data_params_schema(data_id=data_id)

    def open_data(self,
                  data_id: str,
                  opener_id: str = None,
                  **open_params) -> xr.Dataset:
        self._assert_valid_data_id(data_id)
        if not opener_id:
            opener_id = self._get_opener_id(data_id)
        path = self._resolve_data_id_to_path(data_id)
        return self._new_s3_opener(opener_id).open_data(data_id=path, **open_params)

    def get_data_writer_ids(self, type_id: str = None) -> Tuple[str, ...]:
        if type_id:
            type_id = TypeId.normalize(type_id)
        if type_id == TYPE_ID_ANY:
            type_id = None
        self._assert_valid_type_id(type_id)
        extensions = find_data_writer_extensions(
            predicate=get_data_accessor_predicate(type_id=type_id, storage_id=_STORAGE_ID)
        )
        return tuple(ext.name for ext in extensions)

    def get_write_data_params_schema(self, writer_id: str = None) -> JsonObjectSchema:
        if not writer_id:
            extensions = find_data_writer_extensions(
                predicate=get_data_accessor_predicate(type_id='dataset', storage_id=_STORAGE_ID)
            )
            writer_id = extensions[0].name
        return self._new_s3_writer(writer_id).get_write_data_params_schema()

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
                                                        storage_id=_STORAGE_ID)
            else:
                raise DataStoreError(f'Unsupported data type {type(data)}')
            extensions = find_data_writer_extensions(predicate=predicate)
            writer_id = extensions[0].name
        data_id = self._ensure_valid_data_id(data_id, data)
        path = self._resolve_data_id_to_path(data_id)
        self._new_s3_writer(writer_id).write_data(data, data_id=path, replace=replace, **write_params)
        self.register_data(data_id, data)
        return data_id

    def delete_data(self, data_id: str):
        path = self._resolve_data_id_to_path(data_id)
        try:
            self._s3_fs.delete(path, recursive=True)
            self.deregister_data(data_id)
        except ValueError as e:
            raise DataStoreError(f'{e}') from e

    def register_data(self, data_id: str, data: Any):
        # TODO: implement me
        pass

    def deregister_data(self, data_id: str):
        # TODO: implement me
        pass

    ###############################################################
    # Implementation helpers

    def _new_s3_opener(self, opener_id):
        self._assert_not_closed()
        return new_data_opener(opener_id, s3_fs=self._s3_fs)

    def _new_s3_writer(self, writer_id):
        self._assert_not_closed()
        return new_data_writer(writer_id, s3_fs=self._s3_fs)

    @classmethod
    def _ensure_valid_data_id(cls, data_id: Optional[str], data: Any) -> str:
        return data_id or str(uuid.uuid4()) + cls._get_filename_ext(data)

    def _assert_not_closed(self):
        if self._s3_fs is None:
            raise DataStoreError(f'Data store already closed.')

    def _assert_valid_data_id(self, data_id):
        if not self.has_data(data_id):
            raise DataStoreError(f'Data resource "{data_id}" does not exist in store')

    def _resolve_data_id_to_path(self, data_id: str) -> str:
        assert_given(data_id, 'data_id')
        return f'{self._bucket_name}/{data_id}'

    def _assert_valid_type_id(self, type_id: Optional[TypeId]):
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
        accessor_id_parts = self._get_accessor_id_parts(data_id, require=require)
        if not accessor_id_parts:
            return []
        type_id, format_id, storage_id = accessor_id_parts
        # print(type_id, format_id, storage_id)
        predicate = get_data_accessor_predicate(type_id=type_id, format_id=format_id, storage_id=storage_id)
        extensions = get_data_accessor_extensions(predicate)
        if not extensions:
            if require:
                raise DataStoreError(f'No accessor found for data resource "{data_id}"')
            return []
        return extensions

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
        type_id = get_type_id(data)
        return _TYPE_ID_TO_ACCESSOR_TO_DEFAULT_FILENAME_EXT[type_id]
