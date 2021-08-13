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

import json
import os.path
import uuid
import warnings
from typing import Optional, Iterator, Any, Tuple, List, Dict, Union, Container

import s3fs
import xarray as xr

from xcube.core.mldataset import MultiLevelDataset
from xcube.core.store import DataDescriptor
from xcube.core.store import DataStoreError
from xcube.core.store import DefaultSearchMixin
from xcube.core.store import MutableDataStore
from xcube.core.store import TYPE_SPECIFIER_ANY
from xcube.core.store import TYPE_SPECIFIER_DATASET
from xcube.core.store import TYPE_SPECIFIER_MULTILEVEL_DATASET
from xcube.core.store import TypeSpecifier
from xcube.core.store import find_data_opener_extensions
from xcube.core.store import find_data_writer_extensions
from xcube.core.store import get_data_accessor_predicate
from xcube.core.store import get_type_specifier
from xcube.core.store import new_data_descriptor
from xcube.core.store import new_data_opener
from xcube.core.store import new_data_writer
from xcube.core.store.accessors.dataset import S3Mixin
from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_in
from xcube.util.assertions import assert_instance
from xcube.util.assertions import assert_true
from xcube.util.extension import Extension
from xcube.util.jsonschema import JsonObjectSchema

_STORAGE_ID = 's3'

_DEFAULT_FORMAT_ID = 'zarr'

_FILENAME_EXT_TO_ACCESSOR_ID_PARTS = {
    '.zarr': (str(TYPE_SPECIFIER_DATASET), 'zarr', _STORAGE_ID),
    '.levels': (str(TYPE_SPECIFIER_MULTILEVEL_DATASET), 'levels', _STORAGE_ID),
}

_FORMAT_TO_FILENAME_EXT = {
    'zarr': '.zarr',
    'levels': '.levels'
}

_REGISTRY_FILE = 'registry.json'


# TODO: write tests
# TODO: complete docs
# TODO: implement '*.levels' support
# TODO: remove code duplication with ./directory.py and its tests.
#   - Introduce a file-system-abstracting base class or mixin, see module "fsspec" and impl. "s3fs" as  used in Dask!
#   - Introduce something like MultiOpenerStoreMixin/MultiWriterStoreMixin!


class S3DataStore(DefaultSearchMixin, MutableDataStore):
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
        self._bucket_name, store_params = \
            S3Mixin.consume_bucket_name_param(store_params)
        assert_given(self._bucket_name, 'bucket_name')
        self._s3, store_params = \
            S3Mixin.consume_s3fs_params(store_params,
                                        check_path=self._bucket_name,
                                        params_name='store_params')
        assert_true(not store_params,
                    'Unknown keyword arguments:'
                    f' {", ".join(store_params.keys())}')
        registry_path = f'{self._bucket_name}/{_REGISTRY_FILE}'
        self._registry = {}
        if self._s3.exists(registry_path):
            with self._s3.open(registry_path, 'r') as registry_file:
                self._registry = json.load(registry_file)

    def close(self):
        pass

    @property
    def s3(self) -> s3fs.S3FileSystem:
        return self._s3

    @property
    def bucket_name(self) -> str:
        return self._bucket_name

    #############################################################################
    # MutableDataStore impl.

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        schema = S3Mixin.get_s3_params_schema()
        schema.required.add('bucket_name')
        return schema

    @classmethod
    def get_type_specifiers(cls) -> Tuple[str, ...]:
        return str(TYPE_SPECIFIER_DATASET),

    def get_type_specifiers_for_data(self, data_id: str) -> Tuple[str, ...]:
        if not self.has_data(data_id):
            raise DataStoreError(f'"{data_id}" is not provided by this data store')
        data_type_specifier, _, _ = self._get_accessor_id_parts(data_id)
        return data_type_specifier,

    def get_data_ids(self,
                     type_specifier: str = None,
                     include_attrs: Container[str] = None) -> \
            Union[Iterator[str], Iterator[Tuple[str, Dict[str, Any]]]]:
        # TODO: do not ignore type_specifier
        # TODO: do not ignore names in include_attrs
        return_tuples = include_attrs is not None
        prefix = self._bucket_name + '/'
        first_index = len(prefix)
        for item in self._s3.listdir(self._bucket_name, detail=False, refresh=True):
            if item.startswith(prefix):
                data_id = item[first_index:]
                yield (data_id, {}) if return_tuples else data_id

    def has_data(self, data_id: str, type_specifier: str = None) -> bool:
        if data_id in self._registry:
            descriptor = self._registry[data_id]
            return not (type_specifier and not TypeSpecifier.normalize(type_specifier).
                        is_satisfied_by(descriptor.type_specifier))
        if type_specifier:
            data_type_specifier, _, _ = self._get_accessor_id_parts(data_id)
            if not TypeSpecifier.parse(data_type_specifier).satisfies(type_specifier):
                return False
        path = self._resolve_data_id_to_path(data_id)
        return self._s3.exists(path)

    def describe_data(self, data_id: str, type_specifier: str = None) -> DataDescriptor:
        if data_id in self._registry:
            descriptor = self._registry[data_id]
            if type_specifier and not TypeSpecifier.normalize(type_specifier). \
                    is_satisfied_by(descriptor.type_specifier):
                raise DataStoreError(f'Data "{data_id}" is not available as '
                                     f'type "{type_specifier}". '
                                     f'It is of type "{descriptor.type_specifier}".')
            return descriptor
        if not self.has_data(data_id, type_specifier):
            msg = f'Data "{data_id}" is not available'
            if not type_specifier:
                raise DataStoreError(msg)
            raise DataStoreError(msg + f' as type "{type_specifier}"')
        data_opener_ids = self.get_data_opener_ids(data_id, type_specifier=type_specifier)
        if len(data_opener_ids) == 0:
            msg = f'Cannot describe data {data_id}, no opener found'
            if not type_specifier:
                raise DataStoreError(msg)
            raise DataStoreError(msg + f' for type "{type_specifier}"')
        open_params = {}
        op_schema = self._new_s3_opener(data_opener_ids[0]).get_open_data_params_schema(data_id)
        if 'consolidated' in op_schema.properties:
            open_params['consolidated'] = True
        data = self.open_data(data_id, data_opener_ids[0], **open_params)
        return new_data_descriptor(data_id, data)

    def get_data_opener_ids(self, data_id: str = None, type_specifier: Optional[str] = None) -> Tuple[str, ...]:
        if type_specifier:
            type_specifier = TypeSpecifier.parse(type_specifier)
        if type_specifier == TYPE_SPECIFIER_ANY:
            type_specifier = None
        self._assert_valid_type_specifier(type_specifier)
        if data_id:
            accessor_id_parts = self._get_accessor_id_parts(data_id, require=False)
            if not accessor_id_parts:
                return tuple()
            acc_type_specifier, acc_format, acc_storage_id = accessor_id_parts
            if not type_specifier:
                type_specifier = acc_type_specifier
            return tuple(ext.name for ext in find_data_opener_extensions(
                predicate=get_data_accessor_predicate(
                    type_specifier=type_specifier,
                    format_id=acc_format,
                    storage_id=acc_storage_id)
            ))
        return tuple(ext.name for ext in find_data_opener_extensions(
            predicate=get_data_accessor_predicate(type_specifier=type_specifier,
                                                  storage_id=_STORAGE_ID)
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

    def get_data_writer_ids(self, type_specifier: str = None) -> Tuple[str, ...]:
        if type_specifier:
            type_specifier = TypeSpecifier.normalize(type_specifier)
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
                predicate=get_data_accessor_predicate(type_specifier='dataset', storage_id=_STORAGE_ID)
            )
            writer_id = extensions[0].name
        return self._new_s3_writer(writer_id).get_write_data_params_schema()

    def write_data(self,
                   data: Any,
                   data_id: str = None,
                   writer_id: str = None,
                   replace: bool = False,
                   **write_params) -> str:
        assert_instance(data, (xr.Dataset, MultiLevelDataset))
        if writer_id and writer_id.split(':')[2] != _STORAGE_ID:
            raise DataStoreError(f'Invalid writer id "{writer_id}"')
        if data_id:
            accessor_id_parts = self._get_accessor_id_parts(data_id, require=False)
            if accessor_id_parts:
                data_type_specifier = get_type_specifier(data)
                if not data_type_specifier.satisfies(accessor_id_parts[0]):
                    raise DataStoreError(f'Data id "{data_id}" is not suitable for data of type '
                                         f'"{data_type_specifier}".')
                predicate = get_data_accessor_predicate(type_specifier=accessor_id_parts[0],
                                                        format_id=accessor_id_parts[1],
                                                        storage_id=accessor_id_parts[2])
                extensions = find_data_writer_extensions(predicate=predicate)
                assert extensions
                writer_id_from_data_id = extensions[0].name
                if writer_id is not None:
                    # checking that a user-provided data id is suitable for the data type and
                    # requested format
                    if writer_id_from_data_id.split(':')[0] != writer_id.split(':')[0] or \
                            writer_id_from_data_id.split(':')[1] != writer_id.split(':')[1]:
                        raise DataStoreError(f'Writer ID "{writer_id}" seems inappropriate for '
                                             f'data id "{data_id}". Try writer id '
                                             f'"{writer_id_from_data_id}" instead.')
                else:
                    writer_id = writer_id_from_data_id
        if not writer_id:
            if isinstance(data, MultiLevelDataset):
                predicate = get_data_accessor_predicate(type_specifier=TYPE_SPECIFIER_MULTILEVEL_DATASET,
                                                        format_id='levels',
                                                        storage_id=_STORAGE_ID)
            elif isinstance(data, xr.Dataset):
                predicate = get_data_accessor_predicate(type_specifier=TYPE_SPECIFIER_DATASET,
                                                        format_id='zarr',
                                                        storage_id=_STORAGE_ID)
            else:
                raise DataStoreError(f'Unsupported data type "{type(data)}"')
            extensions = find_data_writer_extensions(predicate=predicate)
            assert extensions
            writer_id = extensions[0].name
        data_format = writer_id.split(':')[1]
        data_id = self._ensure_valid_data_id(data_format, data_id)
        path = self._resolve_data_id_to_path(data_id)
        self._new_s3_writer(writer_id).write_data(data, data_id=path, replace=replace, **write_params)
        self.register_data(data_id, data)
        return data_id

    def delete_data(self, data_id: str):
        path = self._resolve_data_id_to_path(data_id)
        try:
            self._s3.delete(path, recursive=True)
            self.deregister_data(data_id)
        except ValueError as e:
            raise DataStoreError(f'{e}') from e

    def register_data(self, data_id: str, data: Any):
        descriptor = new_data_descriptor(data_id, data)
        self._registry[data_id] = descriptor
        self._maybe_update_json_registry()

    def deregister_data(self, data_id: str):
        if data_id in self._registry:
            self._registry.pop(data_id)
            self._maybe_update_json_registry()

    def _maybe_update_json_registry(self):
        registry_path = f'{self._bucket_name}/{_REGISTRY_FILE}'
        if self._s3.exists(registry_path):
            with self._s3.open(registry_path, 'w') as registry_file:
                json.dump(self._registry, registry_file)

    ###############################################################
    # Implementation helpers

    def _new_s3_opener(self, opener_id):
        self._assert_not_closed()
        return new_data_opener(opener_id, s3=self._s3)

    def _new_s3_writer(self, writer_id):
        self._assert_not_closed()
        return new_data_writer(writer_id, s3=self._s3)

    @classmethod
    def _ensure_valid_data_id(cls, data_format: str, data_id: Optional[str]) -> str:
        extension = _FORMAT_TO_FILENAME_EXT[data_format]
        if data_id is not None:
            if not data_id.endswith(extension):
                warnings.warn(f'Data if "{data_id}" has no expected file extension.'
                              f'It will be written as "{data_format}".'
                              f'You may encounter issues with accessing the data from the store.')
            return data_id
        return str(uuid.uuid4()) + extension

    def _assert_not_closed(self):
        if self._s3 is None:
            raise DataStoreError(f'Data store already closed.')

    def _assert_valid_data_id(self, data_id):
        if not self.has_data(data_id):
            raise DataStoreError(f'Data resource "{data_id}" does not exist in store')

    def _resolve_data_id_to_path(self, data_id: str) -> str:
        assert_given(data_id, 'data_id')
        return f'{self._bucket_name}/{data_id}'

    def _assert_valid_type_specifier(self, type_specifier: Optional[TypeSpecifier]):
        if type_specifier:
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

    @classmethod
    def _get_accessor_id_parts(cls, data_id: str, require=True) -> Optional[Tuple[str, str, str]]:
        assert_given(data_id, 'data_id')
        _, ext = os.path.splitext(data_id)
        accessor_id_parts = _FILENAME_EXT_TO_ACCESSOR_ID_PARTS.get(ext)
        if not accessor_id_parts and require:
            raise DataStoreError(f'A dataset named "{data_id}" is not supported')
        return accessor_id_parts

    @classmethod
    def _get_filename_ext(cls, data: Any) -> Optional[str]:
        type_specifier = get_type_specifier(data)
        if TYPE_SPECIFIER_MULTILEVEL_DATASET.satisfies(type_specifier):
            return '.levels'
        return '.zarr'
