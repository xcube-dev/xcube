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
import uuid
import warnings
from threading import Lock
from typing import Optional, Iterator, Any, Tuple, List, Dict, Union, Container, Callable

import fsspec
import geopandas as gpd
import xarray as xr

from xcube.core.mldataset import MultiLevelDataset
from xcube.core.store import DataDescriptor
from xcube.core.store import DataOpener
from xcube.core.store import DataStoreError
from xcube.core.store import DataWriter
from xcube.core.store import MutableDataStore
from xcube.core.store import TYPE_SPECIFIER_ANY
from xcube.core.store import TYPE_SPECIFIER_DATASET
from xcube.core.store import TYPE_SPECIFIER_GEODATAFRAME
from xcube.core.store import TYPE_SPECIFIER_MULTILEVEL_DATASET
from xcube.core.store import TypeSpecifier
from xcube.core.store import find_data_opener_extensions
from xcube.core.store import find_data_writer_extensions
from xcube.core.store import get_data_accessor_predicate
from xcube.core.store import get_type_specifier
from xcube.core.store import new_data_descriptor
from xcube.core.store import new_data_opener
from xcube.core.store import new_data_writer
from xcube.core.store.fs.common import FS_PARAMS_PARAM_NAME
from xcube.core.store.fs.common import FileFsAccessor
from xcube.core.store.fs.common import FsAccessor
from xcube.core.store.fs.common import MemoryFsAccessor
from xcube.core.store.fs.common import S3FsAccessor
from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_in
from xcube.util.assertions import assert_instance
from xcube.util.extension import Extension
from xcube.util.jsonschema import JsonBooleanSchema, JsonIntegerSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema

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


class BaseFsDataStore(MutableDataStore):
    """
    Base class for data stores that use an underlying file system
    (fs.AbstractFileSystem).

    :param fs: Optional file system. If not given,
        :meth:_get_fs() should return a file system instance.
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
        self._root = root or ''
        self._max_depth = max_depth
        self._read_only = read_only
        self._fs = fs
        self._lock = Lock()

    @property
    def fs_protocol(self) -> str:
        """
        Get the file system protocol.
        Should be overridden by clients, as the default
        implementation instantiates the file system.
        """
        return self.fs.protocol

    @property
    def fs(self) -> fsspec.AbstractFileSystem:
        """
        Return the underlying file system .
        :return: An instance of ``fs.AbstractFileSystem``.
        """
        if self._fs is None:
            # Lazily instantiate file system.
            with self._lock:
                self._fs = self._load_fs()
        return self._fs

    def _load_fs(self) -> fsspec.AbstractFileSystem:
        """
        Get an instance of the underlying file system.
        :return: An instance of ``fs.AbstractFileSystem``.
        """
        raise NotImplementedError()

    @property
    def root(self) -> str:
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
        actual_type_specifier, _, _ = self._get_accessor_id_parts(data_id)
        return actual_type_specifier,

    def get_data_ids(self,
                     type_specifier: str = None,
                     include_attrs: Container[str] = None) -> \
            Union[Iterator[str], Iterator[Tuple[str, Dict[str, Any]]]]:
        type_specifier = TypeSpecifier.normalize(type_specifier)
        # TODO: do not ignore names in include_attrs
        return_tuples = include_attrs is not None
        # TODO: Use self.fs.walk() to support self.max_depth
        # os.listdir does not guarantee any ordering of the entries, so
        # sort them to ensure predictable behaviour.
        potential_data_ids = sorted(self.fs.listdir(self._root))
        for data_id in potential_data_ids:
            if self._is_data_specified(data_id, type_specifier):
                yield (data_id, {}) if return_tuples else data_id

    def has_data(self, data_id: str, type_specifier: str = None) -> bool:
        assert_given(data_id, 'data_id')
        if self._is_data_specified(data_id, type_specifier):
            path = self._resolve_data_id_to_path(data_id)
            return self.fs.exists(path)
        return False

    def describe_data(self, data_id: str, type_specifier: str = None) \
            -> DataDescriptor:
        self._assert_valid_data_id(data_id)
        self._assert_data_specified(data_id, type_specifier)
        data = self.open_data(data_id)
        return new_data_descriptor(data_id, data, require=True)

    def search_data(self,
                    type_specifier: str = None,
                    **search_params) \
            -> Iterator[DataDescriptor]:
        # TODO: implement me!
        raise NotImplementedError(
            f'search_data() is not yet implemented'
            f' for {BaseFsDataStore}'
        )

    def get_data_opener_ids(self,
                            data_id: str = None,
                            type_specifier: Optional[str] = None) \
            -> Tuple[str, ...]:
        type_specifier = TypeSpecifier.normalize(type_specifier)
        format_id = None
        storage_id = self.fs_protocol
        if data_id:
            accessor_id_parts = self._get_accessor_id_parts(
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
        schema = opener.get_open_data_params_schema(data_id=data_id)
        return self._strip_data_accessor_params_schema(schema)

    def open_data(self,
                  data_id: str,
                  opener_id: str = None,
                  **open_params) -> xr.Dataset:
        opener = self._find_opener(opener_id=opener_id, data_id=data_id)
        path = self._resolve_data_id_to_path(data_id)
        return opener.open_data(path, fs=self.fs, **open_params)

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
        schema = writer.get_write_data_params_schema()
        return self._strip_data_accessor_params_schema(schema)

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
        data_id = self._ensure_valid_data_id(writer_id, data_id=data_id)
        path = self._resolve_data_id_to_path(data_id)
        writer.write_data(data,
                          path,
                          replace=replace,
                          fs=self.fs,
                          **write_params)
        return data_id

    def get_delete_data_params_schema(self, data_id: str = None) \
            -> JsonObjectSchema:
        writer = self._find_writer(data_id=data_id)
        schema = writer.get_delete_data_params_schema(data_id)
        return self._strip_data_accessor_params_schema(schema)

    def delete_data(self, data_id: str, **delete_params):
        if self.read_only:
            raise DataStoreError('Data store is read-only.')
        writer = self._find_writer(data_id=data_id)
        path = self._resolve_data_id_to_path(data_id)
        writer.delete_data(path,
                           fs=self.fs,
                           **delete_params)

    def register_data(self, data_id: str, data: Any):
        # We don't need this as we use the file system
        pass

    def deregister_data(self, data_id: str):
        # We don't need this as we use the file system
        pass

    ###############################################################
    # Implementation helpers

    def _guess_writer_id(self, data, data_id: str = None):
        type_specifier = None
        format_id = None
        if data_id:
            accessor_id_parts = self._get_accessor_id_parts(
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
        actual_type_specifier = self._get_type_specifier_for_data_id(
            data_id, require=False
        )
        if actual_type_specifier is None:
            if require:
                raise DataStoreError(f'Data resource "{data_id}" not found')
            return False
        if not actual_type_specifier.satisfies(type_specifier):
            if require:
                raise DataStoreError(f'Type specifier "{type_specifier}"'
                                     f' cannot be satisfied'
                                     f' by type specifier'
                                     f' "{actual_type_specifier}" of'
                                     f' data resource "{data_id}"')
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

    def _resolve_data_id_to_path(self, data_id: str) -> str:
        assert_given(data_id, 'data_id')
        return f'{self._root}/{data_id}'

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
            accessor_id_parts = self._get_accessor_id_parts(
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

    def _get_type_specifier_for_data_id(self, data_id: str, require=True) \
            -> Optional[TypeSpecifier]:
        accessor_id_parts = self._get_accessor_id_parts(data_id,
                                                        require=require)
        if accessor_id_parts is None:
            return None
        actual_type_specifier, _, _ = accessor_id_parts
        return TypeSpecifier.parse(actual_type_specifier)

    def _get_accessor_id_parts(self, data_id: str, require=True) \
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

    @classmethod
    def _get_filename_ext_for_data(cls, data: Any) -> Optional[str]:
        type_specifier = get_type_specifier(data)
        if type_specifier is None:
            return None
        if TYPE_SPECIFIER_MULTILEVEL_DATASET.is_satisfied_by(type_specifier):
            return '.levels'
        if TYPE_SPECIFIER_GEODATAFRAME.is_satisfied_by(type_specifier):
            return '.geojson'
        return '.zarr'

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


class DeferredFsDataStore(BaseFsDataStore):
    """
    Specialization of :class:BaseFsDataStore
    which defers instantiation of the filesystem until needed.

    :param fs_protocol: The file system protocol,
        e.g. "file", "s3", "gfs".
    :param fs_params: Optional parameters specific for the file system
        identified by *fs_protocol*.
    :param root: Root or base directory.
        Defaults to "".
    :param max_depth: Maximum recursion depth. None means limitless.
        Defaults to 1.
    :param read_only: Whether this is a read-only store.
        Defaults to False.
    """

    def __init__(self,
                 fs_protocol: str,
                 fs_params: Dict[str, Any] = None,
                 root: str = '',
                 max_depth: Optional[int] = 1,
                 read_only: bool = False):
        assert_given(fs_protocol, name='fs_protocol')
        self._fs_protocol = fs_protocol
        self._fs_params = fs_params or {}
        super().__init__(root=root,
                         max_depth=max_depth,
                         read_only=read_only)

    @property
    def fs_protocol(self) -> str:
        # Base class returns self.fs.protocol which
        # instantiates the file system.
        # Avoid this, as we know the protocol up front.
        return self._fs_protocol

    @property
    def fs_params(self) -> Dict[str, Any]:
        return self._fs_params

    def _load_fs(self) -> fsspec.AbstractFileSystem:
        return fsspec.filesystem(self._fs_protocol, **self._fs_params)


class FsDataStore(BaseFsDataStore, FsAccessor):
    """
    Specialization of a :class:BaseFsDataStore that
    also implements a :class:FsAccessor which serves
    the file system.

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
        # instantiates the file system.
        # Avoid this, as we know the protocol up front.
        return self.get_fs_protocol()

    @property
    def fs_params(self) -> Dict[str, Any]:
        return self._fs_params

    def _load_fs(self) -> fsspec.AbstractFileSystem:
        fs, _ = self.get_fs({FS_PARAMS_PARAM_NAME: self._fs_params})
        return fs

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        schema = super().get_data_store_params_schema()
        schema = copy.deepcopy(schema)
        schema.properties[FS_PARAMS_PARAM_NAME] = cls.get_fs_params_schema()
        return schema


class FileFsDataStore(FsDataStore, FileFsAccessor):
    """
    A data store that uses the local filesystem.
    """


class S3FsDataStore(FsDataStore, S3FsAccessor):
    """
    A data store that uses AWS S3 compatible object storage.
    """


class MemoryFsDataStore(FsDataStore, MemoryFsAccessor):
    """
    A data store that uses an in-memory filesystem.
    """
