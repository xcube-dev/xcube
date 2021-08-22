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

import copy
import os.path
import uuid
import warnings
from threading import RLock
from typing import Optional, Iterator, Any, Tuple, List, Dict, \
    Union, Container, Callable

import fsspec
import geopandas as gpd
import xarray as xr

from xcube.core.mldataset import MultiLevelDataset
from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_in
from xcube.util.assertions import assert_instance
from xcube.util.extension import Extension
from xcube.util.jsonschema import JsonBooleanSchema, JsonIntegerSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema
from .accessor import FS_PARAMS_PARAM_NAME
from .accessor import FsAccessor
from .helpers import is_local_fs
from .helpers import normalize_path
from ..accessor import DataOpener
from ..accessor import DataWriter
from ..accessor import find_data_opener_extensions
from ..accessor import find_data_writer_extensions
from ..accessor import get_data_accessor_predicate
from ..accessor import new_data_opener
from ..accessor import new_data_writer
from ..assertions import assert_valid_params
from ..descriptor import DataDescriptor
from ..descriptor import new_data_descriptor
from ..error import DataStoreError
from ..store import MutableDataStore
from ..typespecifier import TYPE_SPECIFIER_ANY
from ..typespecifier import TYPE_SPECIFIER_DATASET
from ..typespecifier import TYPE_SPECIFIER_GEODATAFRAME
from ..typespecifier import TYPE_SPECIFIER_MULTILEVEL_DATASET
from ..typespecifier import TypeSpecifier
from ..search import DefaultSearchMixin

_DEFAULT_TYPE_SPECIFIER = 'dataset'
_DEFAULT_FORMAT_ID = 'zarr'

_FILENAME_EXT_TO_TYPE_SPECIFIER_STR = {
    '.zarr': str(TYPE_SPECIFIER_DATASET),
    '.levels': str(TYPE_SPECIFIER_MULTILEVEL_DATASET),
    '.nc': str(TYPE_SPECIFIER_DATASET),
    '.shp': str(TYPE_SPECIFIER_GEODATAFRAME),
    '.geojson': str(TYPE_SPECIFIER_GEODATAFRAME),
}

_TYPE_SPECIFIER_STR_TO_FILENAME = {
    v: k for k, v in _FILENAME_EXT_TO_TYPE_SPECIFIER_STR.items()
}

_FILENAME_EXT_SET = set(_FILENAME_EXT_TO_TYPE_SPECIFIER_STR.keys())

_FILENAME_EXT_TO_FORMAT = {
    '.zarr': 'zarr',
    '.levels': 'levels',
    '.nc': 'netcdf',
    '.shp': 'shapefile',
    '.geojson': 'geojson',
}

_FORMAT_TO_FILENAME_EXT = {
    v: k for k, v in _FILENAME_EXT_TO_FORMAT.items()
}


class BaseFsDataStore(DefaultSearchMixin, MutableDataStore):
    """
    Base class for data stores that use an underlying filesystem
    of type ``fsspec.AbstractFileSystem``.

    :param fs: Optional filesystem. If not given,
        :meth:_load_fs() must return a filesystem instance.
    :param root: Root or base directory.
        Defaults to "".
    :param max_depth: Maximum recursion depth. None means limitless.
        Defaults to 1.
    :param read_only: Whether this is a read-only store.
        Defaults to False.
    """

    def __init__(self,
                 fs: Optional[fsspec.AbstractFileSystem] = None,
                 root: str = '',
                 max_depth: Optional[int] = 1,
                 read_only: bool = False):
        if fs is not None:
            assert_instance(fs, fsspec.AbstractFileSystem, name='fs')
        self._fs = fs
        self._raw_root: str = root or ''
        self._root: Optional[str] = None
        self._max_depth = max_depth
        self._read_only = read_only
        self._lock = RLock()

    @property
    def fs_protocol(self) -> str:
        """
        Get the filesystem protocol.
        Should be overridden by clients, because the default
        implementation instantiates the filesystem.
        """
        protocol = self.fs.protocol
        return protocol if isinstance(protocol, str) else protocol[0]

    @property
    def fs(self) -> fsspec.AbstractFileSystem:
        """
        Return the underlying filesystem .
        :return: An instance of ``fsspec.AbstractFileSystem``.
        """
        if self._fs is None:
            # Lazily instantiate filesystem.
            with self._lock:
                self._fs = self._load_fs()
        return self._fs

    def _load_fs(self) -> fsspec.AbstractFileSystem:
        """
        Get an instance of the underlying filesystem.
        Default implementation raises NotImplementedError.
        :return: An instance of ``fsspec.AbstractFileSystem``.
        """
        raise NotImplementedError('unknown filesystem')

    @property
    def root(self) -> str:
        if self._root is None:
            # Lazily instantiate root path.
            is_local = is_local_fs(self.fs)
            with self._lock:
                root = self._raw_root
                if is_local:
                    root = os.path.abspath(root)
                self._root = normalize_path(root)
        return self._root

    @property
    def max_depth(self) -> Optional[int]:
        return self._max_depth

    @property
    def read_only(self) -> bool:
        return self._read_only

    #########################################################################
    # MutableDataStore impl.

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                root=JsonStringSchema(default=''),
                max_depth=JsonIntegerSchema(nullable=True, default=1),
                read_only=JsonBooleanSchema(default=False),
            ),
            additional_properties=False
        )

    @classmethod
    def get_type_specifiers(cls) -> Tuple[str, ...]:
        return tuple(_TYPE_SPECIFIER_STR_TO_FILENAME.keys())

    def get_type_specifiers_for_data(self, data_id: str) -> Tuple[str, ...]:
        self._assert_valid_data_id(data_id)
        actual_type_specifier, _, _ = self._guess_accessor_id_parts(data_id)
        return actual_type_specifier,

    def get_data_ids(self,
                     type_specifier: str = None,
                     include_attrs: Container[str] = None) -> \
            Union[Iterator[str], Iterator[Tuple[str, Dict[str, Any]]]]:
        type_specifier = TypeSpecifier.normalize(type_specifier)
        # TODO: do not ignore names in include_attrs
        return_tuples = include_attrs is not None

        potential_data_ids = sorted(list(map(
            self._convert_fs_path_into_data_id,
            self.fs.find(self.root,
                         maxdepth=self._max_depth,
                         withdirs=True)
        )))
        for data_id in potential_data_ids:
            if self._is_data_specified(data_id, type_specifier):
                yield (data_id, {}) if return_tuples else data_id

    def has_data(self, data_id: str, type_specifier: str = None) -> bool:
        assert_given(data_id, 'data_id')
        if self._is_data_specified(data_id, type_specifier):
            fs_path = self._convert_data_id_into_fs_path(data_id)
            return self.fs.exists(fs_path)
        return False

    def describe_data(self, data_id: str, type_specifier: str = None) \
            -> DataDescriptor:
        self._assert_valid_data_id(data_id)
        self._assert_data_specified(data_id, type_specifier)
        # TODO: optimize me, self.open_data() may be very slow!
        #   For Zarr, try using self.fs to load metadata only
        #   rather than instantiating xr.Dataset instances which
        #   can be very expensive for large Zarrs (xarray 0.18.2),
        #   especially in S3 filesystems.
        data = self.open_data(data_id)
        return new_data_descriptor(data_id, data, require=True)

    def get_data_opener_ids(self,
                            data_id: str = None,
                            type_specifier: Optional[str] = None) \
            -> Tuple[str, ...]:
        type_specifier = TypeSpecifier.normalize(type_specifier)
        format_id = None
        storage_id = self.fs_protocol
        if data_id:
            accessor_id_parts = self._guess_accessor_id_parts(
                data_id, require=False
            )
            if not accessor_id_parts:
                return ()  # nothing found
            acc_type_specifier, format_id, storage_id = accessor_id_parts
            if type_specifier == TYPE_SPECIFIER_ANY:
                type_specifier = acc_type_specifier
        return tuple(ext.name for ext in find_data_opener_extensions(
            predicate=get_data_accessor_predicate(
                type_specifier=type_specifier,
                format_id=format_id,
                storage_id=storage_id
            )
        ))

    def get_open_data_params_schema(self,
                                    data_id: str = None,
                                    opener_id: str = None) \
            -> JsonObjectSchema:
        opener = self._find_opener(opener_id=opener_id, data_id=data_id)
        return self._get_open_data_params_schema(opener, data_id)

    def open_data(self,
                  data_id: str,
                  opener_id: str = None,
                  **open_params) -> xr.Dataset:
        opener = self._find_opener(opener_id=opener_id, data_id=data_id)
        open_params_schema = self._get_open_data_params_schema(opener,
                                                               data_id)
        assert_valid_params(open_params,
                            name='open_params',
                            schema=open_params_schema)
        fs_path = self._convert_data_id_into_fs_path(data_id)
        return opener.open_data(fs_path, fs=self.fs, **open_params)

    def get_data_writer_ids(self, type_specifier: str = None) \
            -> Tuple[str, ...]:
        type_specifier = TypeSpecifier.normalize(type_specifier)
        return tuple(ext.name for ext in find_data_writer_extensions(
            predicate=get_data_accessor_predicate(
                type_specifier=type_specifier,
                storage_id=self.fs_protocol
            )
        ))

    def get_write_data_params_schema(self, writer_id: str = None) \
            -> JsonObjectSchema:
        writer = self._find_writer(writer_id=writer_id)
        return self._get_write_data_params_schema(writer)

    def write_data(self,
                   data: Any,
                   data_id: str = None,
                   writer_id: str = None,
                   replace: bool = False,
                   **write_params) -> str:
        if self.read_only:
            raise DataStoreError('Data store is read-only.')
        if not writer_id:
            writer_id = self._guess_writer_id(data, data_id=data_id)
        writer = self._find_writer(writer_id=writer_id)
        write_params_schema = self._get_write_data_params_schema(writer)
        assert_valid_params(write_params,
                            name='write_params',
                            schema=write_params_schema)
        data_id = self._ensure_valid_data_id(writer_id, data_id=data_id)
        fs_path = self._convert_data_id_into_fs_path(data_id)
        self.fs.makedirs(self.root, exist_ok=True)
        writer.write_data(data,
                          fs_path,
                          replace=replace,
                          fs=self.fs,
                          **write_params)
        return data_id

    def get_delete_data_params_schema(self, data_id: str = None) \
            -> JsonObjectSchema:
        writer = self._find_writer(data_id=data_id)
        return self._get_delete_data_params_schema(writer, data_id)

    def delete_data(self, data_id: str, **delete_params):
        if self.read_only:
            raise DataStoreError('Data store is read-only.')
        writer = self._find_writer(data_id=data_id)
        delete_params_schema = self._get_delete_data_params_schema(writer, data_id)
        assert_valid_params(delete_params,
                            name='delete_params',
                            schema=delete_params_schema)
        fs_path = self._convert_data_id_into_fs_path(data_id)
        writer.delete_data(fs_path,
                           fs=self.fs,
                           **delete_params)

    def register_data(self, data_id: str, data: Any):
        # We don't need this as we use the filesystem
        pass

    def deregister_data(self, data_id: str):
        # We don't need this as we use the filesystem
        pass

    ###############################################################
    # Implementation helpers

    def _get_open_data_params_schema(self, opener: DataOpener,
                                     data_id: str):
        schema = opener.get_open_data_params_schema(data_id=data_id)
        return self._strip_data_accessor_params_schema(schema)

    def _get_write_data_params_schema(self, writer: DataWriter):
        schema = writer.get_write_data_params_schema()
        return self._strip_data_accessor_params_schema(schema)

    def _get_delete_data_params_schema(self, writer: DataWriter,
                                       data_id: str):
        schema = writer.get_delete_data_params_schema(data_id)
        return self._strip_data_accessor_params_schema(schema)

    def _guess_writer_id(self, data, data_id: str = None):
        type_specifier = None
        format_id = None
        if data_id:
            accessor_id_parts = self._guess_accessor_id_parts(
                data_id, require=False
            )
            if accessor_id_parts:
                type_specifier = accessor_id_parts[0]
                format_id = accessor_id_parts[1]
        if isinstance(data, xr.Dataset):
            type_specifier = str(TYPE_SPECIFIER_DATASET)
            format_id = format_id or 'zarr'
        elif isinstance(data, MultiLevelDataset):
            type_specifier = str(TYPE_SPECIFIER_MULTILEVEL_DATASET)
            format_id = format_id or 'levels'
        elif isinstance(data, gpd.GeoDataFrame):
            type_specifier = str(TYPE_SPECIFIER_GEODATAFRAME)
            format_id = format_id or 'geojson'
        predicate = get_data_accessor_predicate(
            type_specifier=type_specifier,
            format_id=format_id,
            storage_id=self.fs_protocol
        )
        extensions = find_data_writer_extensions(predicate=predicate)
        if not extensions:
            raise DataStoreError(f'Can not determine data writer'
                                 f' for data of type {type(data)!r}')
        return extensions[0].name

    def _find_opener(self,
                     opener_id: str = None,
                     data_id: str = None,
                     require: bool = True) -> Optional[DataOpener]:
        if not opener_id:
            opener_id = self._find_opener_id(data_id=data_id, require=require)
            if opener_id is None:
                return None
        return new_data_opener(opener_id)

    def _find_writer(self,
                     writer_id: str = None,
                     data_id: str = None,
                     require: bool = True) -> Optional[DataWriter]:
        if not writer_id:
            writer_id = self._find_writer_id(data_id=data_id, require=require)
            if writer_id is None:
                return None
        return new_data_writer(writer_id)

    @classmethod
    def _strip_data_accessor_params_schema(cls, schema: JsonObjectSchema):
        if FS_PARAMS_PARAM_NAME in schema.properties:
            schema = copy.deepcopy(schema)
            del schema.properties[FS_PARAMS_PARAM_NAME]
        return schema

    def _is_data_specified(self,
                           data_id: str,
                           type_specifier,
                           require: bool = False) -> bool:
        type_specifier = TypeSpecifier.normalize(type_specifier)
        actual_type_specifier = self._guess_type_specifier_for_data_id(
            data_id, require=False
        )
        if actual_type_specifier is None:
            if require:
                raise DataStoreError(f'Cannot determine data type of'
                                     f' resource {data_id!r}')
            return False
        if not actual_type_specifier.satisfies(type_specifier):
            if require:
                raise DataStoreError(f'Data type {type_specifier!r}'
                                     f' is not compatible with type'
                                     f' {actual_type_specifier!r} of'
                                     f' data resource {data_id!r}')
            return False
        return True

    def _assert_data_specified(self, data_id, type_specifier):
        self._is_data_specified(data_id, type_specifier, require=True)

    @classmethod
    def _ensure_valid_data_id(cls, writer_id: str, data_id: str = None) -> str:
        format_id = writer_id.split(':')[1]
        extension = _FORMAT_TO_FILENAME_EXT[format_id]
        if data_id:
            if not data_id.endswith(extension):
                warnings.warn(f'Data resource identifier {data_id!r} is'
                              f' lacking an expected extension {extension!r}.'
                              f' It will be written as {format_id!r},'
                              f' but the store may later have difficulties'
                              f' identifying the correct data format.')
            return data_id
        return str(uuid.uuid4()) + extension

    def _assert_valid_data_id(self, data_id):
        if not self.has_data(data_id):
            raise DataStoreError(f'Data resource "{data_id}"'
                                 f' does not exist in store')

    def _convert_data_id_into_fs_path(self, data_id: str) -> str:
        assert_given(data_id, 'data_id')
        root = self.root
        fs_path = f'{root}/{data_id}' if root else data_id
        return fs_path

    def _convert_fs_path_into_data_id(self, fs_path: str) -> str:
        assert_given(fs_path, 'fs_path')
        fs_path = normalize_path(fs_path)
        root = self.root
        if root:
            root_pos = fs_path.find(root)
            if root_pos == -1:
                raise RuntimeError('internal error')
            data_id = fs_path[root_pos + len(root):]
        else:
            data_id = fs_path
        while data_id.startswith('/'):
            data_id = data_id[1:]
        return data_id

    def _assert_valid_type_specifier(self,
                                     type_specifier: TypeSpecifier):
        if type_specifier != TYPE_SPECIFIER_ANY:
            assert_in(type_specifier,
                      self.get_type_specifiers(),
                      name='type_specifier')

    def _find_opener_id(self, data_id: str = None, require=True) \
            -> Optional[str]:
        return self._find_accessor_id(find_data_opener_extensions,
                                      data_id=data_id,
                                      require=require)

    def _find_writer_id(self, data_id: str = None, require=True) \
            -> Optional[str]:
        return self._find_accessor_id(find_data_writer_extensions,
                                      data_id=data_id,
                                      require=require)

    def _find_accessor_id(self,
                          find_data_accessor_extensions: Callable,
                          data_id: str = None,
                          require=True) -> Optional[str]:
        extensions = self._find_accessor_extensions(
            find_data_accessor_extensions,
            data_id=data_id,
            require=require
        )
        return extensions[0].name if extensions else None

    def _find_opener_extensions(self, data_id: str = None, require=True):
        return self._find_accessor_extensions(find_data_opener_extensions,
                                              data_id=data_id,
                                              require=require)

    def _find_writer_extensions(self, data_id: str = None, require=True):
        return self._find_accessor_extensions(find_data_writer_extensions,
                                              data_id=data_id,
                                              require=require)

    def _find_accessor_extensions(self,
                                  find_data_accessor_extensions: Callable,
                                  data_id: str = None,
                                  require=True) -> List[Extension]:
        if data_id:
            accessor_id_parts = self._guess_accessor_id_parts(
                data_id,
                require=require
            )
            if not accessor_id_parts:
                return []
            type_specifier = accessor_id_parts[0]
            format_id = accessor_id_parts[1]
            storage_id = accessor_id_parts[2]
        else:
            type_specifier = _DEFAULT_TYPE_SPECIFIER
            format_id = _DEFAULT_FORMAT_ID
            storage_id = self.fs_protocol
        predicate = get_data_accessor_predicate(
            type_specifier=type_specifier,
            format_id=format_id,
            storage_id=storage_id
        )
        extensions = find_data_accessor_extensions(predicate)
        if not extensions:
            if require:
                msg = 'No data accessor found'
                if data_id:
                    msg += f' for data resource {data_id!r}'
                raise DataStoreError(msg)
            return []
        return extensions

    def _guess_type_specifier_for_data_id(self, data_id: str, require=True) \
            -> Optional[TypeSpecifier]:
        accessor_id_parts = self._guess_accessor_id_parts(data_id,
                                                          require=require)
        if accessor_id_parts is None:
            return None
        actual_type_specifier, _, _ = accessor_id_parts
        return TypeSpecifier.parse(actual_type_specifier)

    def _guess_accessor_id_parts(self, data_id: str, require=True) \
            -> Optional[Tuple[str, str, str]]:
        assert_given(data_id, 'data_id')
        ext = self._get_filename_ext(data_id)
        type_specifier = _FILENAME_EXT_TO_TYPE_SPECIFIER_STR.get(ext)
        format_name = _FILENAME_EXT_TO_FORMAT.get(ext)
        if type_specifier is None or format is None:
            if require:
                raise DataStoreError(f'A dataset named'
                                     f' "{data_id}" is not supported')
            return None
        return type_specifier, format_name, self.fs_protocol

    def _get_filename_ext(self, data_path: str) -> str:
        dot_pos = data_path.rfind('.')
        if dot_pos == -1:
            return ''
        slash_pos = data_path.rfind('/')
        if dot_pos < slash_pos:
            return ''
        # TODO: avoid self.fs here to ensure laziness.
        if self.fs.sep != '/':
            sep_pos = data_path.rfind(self.fs.sep)
            if dot_pos < sep_pos:
                return ''
        return data_path[dot_pos:]


class FsDataStore(BaseFsDataStore, FsAccessor):
    """
    Specialization of a :class:BaseFsDataStore that
    also implements a :class:FsAccessor which serves
    the filesystem.

    :param fs_params: Parameters specific to the underlying filesystem.
        Used to instantiate the filesystem.
    :param root: Root or base directory.
        Defaults to "".
    :param max_depth: Maximum recursion depth. None means limitless.
        Defaults to 1.
    :param read_only: Whether this is a read-only store.
        Defaults to False.
    """

    def __init__(self,
                 fs_params: Dict[str, Any] = None,
                 root: str = '',
                 max_depth: Optional[int] = 1,
                 read_only: bool = False):
        self._fs_params = fs_params or {}
        super().__init__(root=root,
                         max_depth=max_depth,
                         read_only=read_only)

    @property
    def fs_protocol(self) -> str:
        # Base class returns self.fs.protocol which
        # instantiates the filesystem.
        # Avoid this, as we know the protocol up front.
        return self.get_fs_protocol()

    @property
    def fs_params(self) -> Dict[str, Any]:
        return self._fs_params

    def _load_fs(self) -> fsspec.AbstractFileSystem:
        # Note, this is invoked only once per store instance.
        fs, _ = self.load_fs({FS_PARAMS_PARAM_NAME: self._fs_params})
        return fs

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        schema = super().get_data_store_params_schema()
        schema = copy.deepcopy(schema)
        schema.properties[FS_PARAMS_PARAM_NAME] = cls.get_fs_params_schema()
        return schema
