# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import copy
import fnmatch
import os.path
import pathlib
import uuid
import warnings
from collections.abc import Container, Iterator, Mapping, Sequence
from threading import RLock
from typing import Any, Callable, Literal, Optional, Tuple, Union

import fsspec
import geopandas as gpd
import xarray as xr

from xcube.core.mldataset import MultiLevelDataset
from xcube.util.assertions import assert_given, assert_in, assert_instance, assert_true
from xcube.util.extension import Extension
from xcube.util.fspath import is_local_fs
from xcube.util.jsonschema import (
    JsonArraySchema,
    JsonBooleanSchema,
    JsonComplexSchema,
    JsonIntegerSchema,
    JsonNullSchema,
    JsonObjectSchema,
    JsonStringSchema,
)

from ..accessor import (
    DataOpener,
    DataWriter,
    find_data_opener_extensions,
    find_data_writer_extensions,
    get_data_accessor_predicate,
    new_data_opener,
    new_data_writer,
)
from ..assertions import assert_valid_params
from ..datatype import (
    ANY_TYPE,
    DATASET_TYPE,
    GEO_DATA_FRAME_TYPE,
    MULTI_LEVEL_DATASET_TYPE,
    DataType,
    DataTypeLike,
)
from ..descriptor import DataDescriptor, new_data_descriptor
from ..error import DataStoreError
from ..search import DefaultSearchMixin
from ..store import MutableDataStore
from .accessor import STORAGE_OPTIONS_PARAM_NAME, FsAccessor

_DEFAULT_DATA_TYPE = DATASET_TYPE.alias
_DEFAULT_FORMAT_ID = "zarr"

_DataId = str
_DataIdTuple = tuple[_DataId, dict[str, Any]]
_DataIdIter = Iterator[_DataId]
_DataIdTupleIter = Iterator[_DataIdTuple]
_DataIds = Union[_DataIdIter, _DataIdTupleIter]


class BaseFsDataStore(DefaultSearchMixin, MutableDataStore):
    """
    Base class for data stores that use an underlying filesystem
    of type ``fsspec.AbstractFileSystem``.

    The data store is capable of filtering the data identifiers reported
    by ``get_data_ids()``. For this purpose the optional keywords
    `excludes` and `includes` are used which can both take the form of
    a wildcard pattern or a sequence of wildcard patterns:

    * ``excludes``: if given and if any pattern matches the identifier,
      the identifier is not reported.
    * ``includes``: if not given or if any pattern matches the identifier,
      the identifier is reported.

    Args:
        fs: Optional filesystem. If not given,
            :meth:`_load_fs` must return a filesystem instance.
        root: Root or base directory.
            Defaults to "".
        max_depth: Maximum recursion depth. None means limitless.
            Defaults to 1.
        read_only: Whether this is a read-only store.
            Defaults to False.
        includes: Optional sequence of wildcards that identify included
            filesystem paths. Affects the data identifiers (paths)
            returned by `get_data_ids()`. By default, all paths are included.
        excludes: Optional sequence of wildcards that identify excluded
            filesystem paths. Affects the data identifiers (paths)
            returned by `get_data_ids()`. By default, no paths are excluded.
    """

    def __init__(
        self,
        fs: Optional[fsspec.AbstractFileSystem] = None,
        root: str = "",
        max_depth: Optional[int] = 1,
        read_only: bool = False,
        includes: Optional[Union[str, Sequence[str]]] = None,
        excludes: Optional[Union[str, Sequence[str]]] = None,
    ):
        if fs is not None:
            assert_instance(fs, fsspec.AbstractFileSystem, name="fs")
        self._fs = fs
        self._raw_root: str = root or ""
        self._root: Optional[str] = None
        self._max_depth = max_depth
        self._read_only = read_only
        self._includes = self._normalize_wc(includes)
        self._excludes = self._normalize_wc(excludes)
        self._filename_ext_to_format_openers = None
        self._filename_ext_to_format_writers = None
        self._format_to_data_type_aliases_openers = None
        self._format_to_data_type_aliases_writers = None
        self._lock = RLock()

    @property
    def protocol(self) -> str:
        """
        Get the filesystem protocol.
        Should be overridden by clients, because the default
        implementation instantiates the filesystem.
        """
        protocol = self.fs.protocol
        return protocol if isinstance(protocol, str) else protocol[0]

    @property
    def fs(self) -> fsspec.AbstractFileSystem:
        """The underlying filesystem.

        Returns: An instance of ``fsspec.AbstractFileSystem``.
        """
        if self._fs is None:
            # Lazily instantiate filesystem.
            with self._lock:
                self._fs = self._load_fs()
        return self._fs

    def _load_fs(self) -> fsspec.AbstractFileSystem:
        """Get an instance of the underlying filesystem.
        Default implementation raises ``NotImplementedError``.

        Returns: An instance of ``fsspec.AbstractFileSystem``.
        Raises: NotImplementedError
        """
        raise NotImplementedError("unknown filesystem")

    @property
    def root(self) -> str:
        if self._root is None:
            # Lazily instantiate root path.
            is_local = is_local_fs(self.fs)
            with self._lock:
                root = self._raw_root
                if is_local:
                    root = os.path.abspath(root)
                self._root = pathlib.Path(root).as_posix()
        return self._root

    @property
    def max_depth(self) -> Optional[int]:
        """Maximum recursion depth. None means limitless."""
        return self._max_depth

    @property
    def read_only(self) -> bool:
        """Whether this is a read-only store."""
        return self._read_only

    @property
    def includes(self) -> tuple[str]:
        """Wildcard patterns that include paths."""
        return self._includes

    @property
    def excludes(self) -> tuple[str]:
        """Wildcard patterns that exclude paths."""
        return self._excludes

    #########################################################################
    # MutableDataStore impl.

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        wc_schema = JsonComplexSchema(
            one_of=[
                JsonNullSchema(),
                JsonStringSchema(),
                JsonArraySchema(items=JsonStringSchema()),
            ]
        )
        return JsonObjectSchema(
            properties=dict(
                root=JsonStringSchema(default=""),
                max_depth=JsonIntegerSchema(nullable=True, default=1),
                read_only=JsonBooleanSchema(default=False),
                includes=wc_schema,
                excludes=wc_schema,
            ),
            additional_properties=False,
        )

    @classmethod
    def get_data_types(cls) -> tuple[str, ...]:
        _, format_to_data_type_aliases = cls._set_infos_from_extensions(
            find_data_opener_extensions
        )
        return tuple(
            sorted({
                data_type
                for types_tuple in format_to_data_type_aliases.values()
                for data_type in types_tuple
            })
        )

    def get_data_types_for_data(self, data_id: str) -> tuple[str, ...]:
        self._assert_valid_data_id(data_id)
        all_opener_id_parts = self._guess_all_opener_id_parts(data_id, require=False)
        data_type_aliases = [dta for dta, format_id, protocol in all_opener_id_parts]
        return tuple(data_type_aliases)

    def get_data_ids(
        self,
        data_type: DataTypeLike = None,
        include_attrs: Container[str] | bool = False,
    ) -> _DataIds:
        data_type = DataType.normalize(data_type)
        # TODO: do not ignore names in include_attrs
        return_tuples = include_attrs is not False
        data_ids = self._generate_data_ids("", data_type, return_tuples, 1)
        if self._includes or self._excludes:
            yield from self._filter_data_ids(data_ids, return_tuples)
        yield from data_ids

    def has_data(self, data_id: str, data_type: DataTypeLike = None) -> bool:
        assert_given(data_id, "data_id")
        if self._is_data_type_available(data_id, data_type):
            fs_path = self._convert_data_id_into_fs_path(data_id)
            if self.protocol == "https":
                fs_path = f"{self.protocol}://{fs_path}"
            return self.fs.exists(fs_path)
        return False

    def describe_data(
        self, data_id: str, data_type: DataTypeLike = None
    ) -> DataDescriptor:
        self._assert_valid_data_id(data_id)
        # TODO: optimize me, self.open_data() may be very slow!
        #   For Zarr, try using self.fs to load metadata only
        #   rather than instantiating xr.Dataset instances which
        #   can be very expensive for large Zarrs (xarray 0.18.2),
        #   especially in S3 filesystems.
        data = self.open_data(data_id)
        return new_data_descriptor(data_id, data, require=True)

    def get_data_opener_ids(
        self, data_id: str = None, data_type: DataTypeLike = None
    ) -> tuple[str, ...]:
        data_type = DataType.normalize(data_type)
        if data_type == ANY_TYPE:
            data_type = None
        if data_id:
            all_opener_id_parts = self._guess_all_opener_id_parts(
                data_id, data_type=data_type, require=False
            )
            results = tuple()
            for acc_data_type_alias, format_id, storage_id in all_opener_id_parts:
                results += tuple(
                    ext.name
                    for ext in find_data_opener_extensions(
                        predicate=get_data_accessor_predicate(
                            data_type=acc_data_type_alias,
                            format_id=format_id,
                            storage_id=storage_id,
                        )
                    )
                )
            return results
        storage_id = self.protocol
        return tuple(
            ext.name
            for ext in find_data_opener_extensions(
                predicate=get_data_accessor_predicate(
                    data_type=data_type, storage_id=storage_id
                )
            )
        )

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        opener = self._find_opener(opener_id=opener_id, data_id=data_id)
        schema = self._get_open_data_params_schema(opener, data_id)
        if opener_id is None:
            # If the schema for a specific opener was requested, we
            # return the opener's schema. Otherwise, we enhance schema
            # for parameters, such as "data_type".
            schema = copy.deepcopy(schema)
            _COMMON_OPEN_DATA_PARAMS_SCHEMA_PROPERTIES = dict(
                data_type=JsonStringSchema(
                    enum=list(self._get_data_types()),
                    title="Optional data type",
                )
            )
            schema.properties |= _COMMON_OPEN_DATA_PARAMS_SCHEMA_PROPERTIES
        return schema

    def open_data(
        self, data_id: str, opener_id: str = None, **open_params
    ) -> xr.Dataset:
        data_type = open_params.pop("data_type", None)
        opener = self._find_opener(
            opener_id=opener_id, data_id=data_id, data_type=data_type
        )
        open_params_schema = self._get_open_data_params_schema(opener, data_id)
        assert_valid_params(open_params, name="open_params", schema=open_params_schema)
        fs_path = self._convert_data_id_into_fs_path(data_id)
        return opener.open_data(fs_path, fs=self.fs, **open_params)

    def get_data_writer_ids(self, data_type: str = None) -> tuple[str, ...]:
        data_type = DataType.normalize(data_type)
        return tuple(
            ext.name
            for ext in find_data_writer_extensions(
                predicate=get_data_accessor_predicate(
                    data_type=data_type, storage_id=self.protocol
                )
            )
        )

    def get_write_data_params_schema(self, writer_id: str = None) -> JsonObjectSchema:
        writer = self._find_writer(writer_id=writer_id)
        return self._get_write_data_params_schema(writer)

    def write_data(
        self,
        data: Any,
        data_id: str = None,
        writer_id: str = None,
        replace: bool = False,
        **write_params,
    ) -> str:
        if self.read_only:
            raise DataStoreError("Data store is read-only.")
        if not writer_id:
            writer_id = self._guess_writer_id(data, data_id=data_id)
        writer = new_data_writer(writer_id)
        write_params_schema = self._get_write_data_params_schema(writer)
        assert_valid_params(
            write_params, name="write_params", schema=write_params_schema
        )
        data_id = self._ensure_valid_data_id(writer_id, data_id=data_id)
        fs_path = self._convert_data_id_into_fs_path(data_id)
        self.fs.makedirs(self.root, exist_ok=True)
        written_fs_path = writer.write_data(
            data, fs_path, replace=replace, fs=self.fs, root=self.root, **write_params
        )
        # Verify, accessors fulfill their write_data() contract
        assert_true(
            fs_path == written_fs_path,
            message="FsDataAccessor implementations must return the data_id passed in.",
        )
        # Return original data_id (which is a relative path).
        # Note: it would be cleaner to return written_fs_path
        # here, but it is an absolute path.
        # --> Possible solution: FsDataAccessor implementations
        # should be responsible for resolving their data_id into a
        # filesystem path by providing them both with fs and root
        # arguments.
        return data_id

    def get_delete_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        writer = self._find_writer(data_id=data_id)
        return self._get_delete_data_params_schema(writer, data_id)

    def delete_data(self, data_id: str, **delete_params):
        if self.read_only:
            raise DataStoreError("Data store is read-only.")
        writer = self._find_writer(data_id=data_id)
        delete_params_schema = self._get_delete_data_params_schema(writer, data_id)
        assert_valid_params(
            delete_params, name="delete_params", schema=delete_params_schema
        )
        fs_path = self._convert_data_id_into_fs_path(data_id)
        writer.delete_data(fs_path, fs=self.fs, root=self.root, **delete_params)

    def register_data(self, data_id: str, data: Any):
        # We don't need this as we use the filesystem
        pass

    def deregister_data(self, data_id: str):
        # We don't need this as we use the filesystem
        pass

    ###############################################################
    # Implementation helpers

    @staticmethod
    def _get_open_data_params_schema(opener: DataOpener, data_id: str):
        schema = opener.get_open_data_params_schema(data_id=data_id)
        return FsAccessor.remove_storage_options_from_params_schema(schema)

    @staticmethod
    def _get_write_data_params_schema(writer: DataWriter):
        schema = writer.get_write_data_params_schema()
        return FsAccessor.remove_storage_options_from_params_schema(schema)

    @staticmethod
    def _get_delete_data_params_schema(writer: DataWriter, data_id: str):
        schema = writer.get_delete_data_params_schema(data_id)
        return FsAccessor.remove_storage_options_from_params_schema(schema)

    def _get_filename_ext_to_format_openers(self):
        if not self._filename_ext_to_format_openers:
            (
                self._filename_ext_to_format_openers,
                self._format_to_data_type_aliases_openers,
            ) = self._set_infos_from_extensions(
                find_data_opener_extensions, self.protocol
            )
        return self._filename_ext_to_format_openers

    def _get_format_to_data_type_aliases_openers(self):
        if not self._format_to_data_type_aliases_openers:
            (
                self._filename_ext_to_format_openers,
                self._format_to_data_type_aliases_openers,
            ) = self._set_infos_from_extensions(
                find_data_opener_extensions, self.protocol
            )
        return self._format_to_data_type_aliases_openers

    def _get_filename_ext_to_format_writers(self):
        if not self._filename_ext_to_format_writers:
            (
                self._filename_ext_to_format_writers,
                self._format_to_data_type_aliases_writers,
            ) = self._set_infos_from_extensions(
                find_data_writer_extensions, self.protocol
            )
        return self._filename_ext_to_format_writers

    def _get_format_to_data_type_aliases_writers(self):
        if not self._format_to_data_type_aliases_writers:
            (
                self._filename_ext_to_format_writers,
                self._format_to_data_type_aliases_writers,
            ) = self._set_infos_from_extensions(
                find_data_writer_extensions, self.protocol
            )
        return self._format_to_data_type_aliases_writers

    @classmethod
    def _set_infos_from_extensions(
        cls, find_extensions, protocol=None
    ) -> (Tuple)[dict, dict]:
        filename_ext_to_format = {}
        format_to_data_type_aliases = {}
        predicate = get_data_accessor_predicate(storage_id=protocol)
        for ext in find_extensions(predicate):
            data_type = ext.name.split(":")[0]
            fmt = ext.name.split(":")[1]
            extensions = ext.metadata.get("extensions", [])
            for extension in extensions:
                filename_ext_to_format[extension] = fmt
            if data_type == "mldataset":
                format_to_data_type_aliases[fmt] = (
                    data_type,
                ) + format_to_data_type_aliases.get(fmt, ())
            else:
                format_to_data_type_aliases[fmt] = format_to_data_type_aliases.get(
                    fmt, ()
                ) + (data_type,)
        return filename_ext_to_format, format_to_data_type_aliases

    def _get_data_types(self):
        format_to_data_type_aliases = self._get_format_to_data_type_aliases_openers()
        return tuple(
            {
                data_type
                for types_tuple in format_to_data_type_aliases.values()
                for data_type in types_tuple
            }
        )

    def _guess_writer_id(self, data, data_id: str = None):
        data_type = None
        format_id = None
        if data_id:
            accessor_id_parts = self._guess_best_writer_id_parts(data_id, require=False)
            if accessor_id_parts:
                data_type = accessor_id_parts[0]
                format_id = accessor_id_parts[1]
        if isinstance(data, xr.Dataset):
            data_type = DATASET_TYPE.alias
            format_id = format_id or "zarr"
        elif isinstance(data, MultiLevelDataset):
            data_type = MULTI_LEVEL_DATASET_TYPE.alias
            format_id = format_id or "levels"
        elif isinstance(data, gpd.GeoDataFrame):
            data_type = GEO_DATA_FRAME_TYPE.alias
            format_id = format_id or "geojson"
        predicate = get_data_accessor_predicate(
            data_type=data_type, format_id=format_id, storage_id=self.protocol
        )
        extensions = find_data_writer_extensions(predicate=predicate)
        if not extensions:
            raise DataStoreError(
                f"Can not find suitable data writer"
                f" for data of type {type(data)!r}"
                f" and format {format_id!r}"
            )
        return extensions[0].name

    def _find_opener(
        self, opener_id: str = None, data_id: str = None, data_type: DataTypeLike = None
    ) -> DataOpener:
        if not opener_id:
            opener_id = self._find_opener_id(
                data_id=data_id, data_type=data_type, require=True
            )
        return new_data_opener(opener_id)

    def _find_writer(self, writer_id: str = None, data_id: str = None) -> DataWriter:
        if not writer_id:
            writer_id = self._find_writer_id(data_id=data_id, require=True)
        return new_data_writer(writer_id)

    def _is_data_type_available(self, data_id: str, data_type: DataTypeLike) -> bool:
        ext = self._get_filename_ext(data_id)
        format_id = self._get_filename_ext_to_format_openers().get(ext.lower())
        if format_id is None:
            return False
        avail_data_types = self._get_format_to_data_type_aliases_openers().get(
            format_id
        )
        data_type = DataType.normalize(data_type)
        return any(
            data_type.is_super_type_of(avail_data_type)
            for avail_data_type in avail_data_types
        )

    def _ensure_valid_data_id(self, writer_id: str, data_id: str = None) -> str:
        format_id = writer_id.split(":")[1]
        first_ext = None
        filename_ext_to_format, _ = self._set_infos_from_extensions(
            find_data_writer_extensions, self.protocol
        )
        for known_ext, known_format_id in filename_ext_to_format.items():
            # Note, there may be multiple common file extensions
            # for a given data format_id, e.g. .tif, .tiff, .geotiff.
            # Must try them all:
            if format_id == known_format_id:
                if first_ext is None:
                    first_ext = known_ext
                if data_id and data_id.endswith(known_ext):
                    return data_id
        assert first_ext is not None
        if data_id:
            warnings.warn(
                f"Data resource identifier {data_id!r} is"
                f" lacking an expected extension {first_ext!r}."
                f" It will be written as {format_id!r},"
                f" but the store may later have difficulties"
                f" identifying the correct data format."
            )
            return data_id
        return str(uuid.uuid4()) + first_ext

    def _assert_valid_data_id(self, data_id):
        if not self.has_data(data_id):
            raise DataStoreError(f'Data resource "{data_id}" does not exist in store')

    def _convert_data_id_into_fs_path(self, data_id: str) -> str:
        assert_given(data_id, "data_id")
        root = self.root
        fs_path = f"{root}/{data_id}" if root else data_id
        return fs_path

    def _assert_valid_data_type(self, data_type: DataType):
        if data_type != ANY_TYPE:
            assert_in(data_type, self.get_data_types(), name="data_type")

    def _find_opener_id(
        self, data_id: str = None, data_type: DataTypeLike = None, require=True
    ) -> Optional[str]:
        return self._find_accessor_id(
            find_data_opener_extensions,
            self._guess_best_opener_id_parts,
            data_id=data_id,
            data_type=data_type,
            require=require,
        )

    def _find_writer_id(self, data_id: str = None, require=True) -> Optional[str]:
        return self._find_accessor_id(
            find_data_writer_extensions,
            self._guess_best_writer_id_parts,
            data_id=data_id,
            require=require,
        )

    def _find_accessor_id(
        self,
        find_data_accessor_extensions: Callable,
        guess_accessor_id_parts: Callable,
        data_id: str = None,
        data_type: DataTypeLike = None,
        require=True,
    ) -> Optional[str]:
        extensions = self._find_accessor_extensions(
            find_data_accessor_extensions,
            guess_accessor_id_parts,
            data_id=data_id,
            data_type=data_type,
            require=require,
        )
        return extensions[0].name if extensions else None

    def _find_accessor_extensions(
        self,
        find_data_accessor_extensions: Callable,
        guess_accessor_id_parts: Callable,
        data_id: str = None,
        data_type: DataTypeLike = None,
        require=True,
    ) -> list[Extension]:
        if data_id:
            accessor_id_parts = guess_accessor_id_parts(
                data_id, data_type=data_type, require=require
            )
            if not accessor_id_parts:
                return []
            data_type_alias = accessor_id_parts[0]
            format_id = accessor_id_parts[1]
            storage_id = accessor_id_parts[2]
        else:
            if data_type:
                data_type_alias = DataType.normalize(data_type).alias
            else:
                data_type_alias = _DEFAULT_DATA_TYPE
            format_id = _DEFAULT_FORMAT_ID
            storage_id = self.protocol

        def _get_extension(type_alias: str) -> list[Extension]:
            predicate = get_data_accessor_predicate(
                data_type=type_alias, format_id=format_id, storage_id=storage_id
            )
            return find_data_accessor_extensions(predicate)

        extensions = _get_extension(data_type_alias)
        if not extensions:
            extensions = _get_extension(_DEFAULT_DATA_TYPE)
            if not extensions:
                if require:
                    msg = "No data accessor found"
                    if data_id:
                        msg += f" for data resource {data_id!r}"
                    raise DataStoreError(msg)
                return []
            else:
                warnings.warn(
                    f"No data opener found for format {format_id!r} and data type "
                    f"{data_type!r}. Data type is changed to the default data type "
                    f"{_DEFAULT_DATA_TYPE!r}."
                )
        return extensions

    def _guess_all_opener_id_parts(
        self, data_id: str, data_type: DataTypeLike = None, require=True
    ) -> list[tuple[str, str, str]]:
        return self._guess_all_accessor_id_parts(
            self._get_filename_ext_to_format_openers(),
            self._get_format_to_data_type_aliases_openers(),
            data_id,
            data_type,
            require,
        )

    def _guess_best_opener_id_parts(
        self, data_id: str, data_type: DataTypeLike = None, require=True
    ) -> Optional[tuple[str, str, str]]:
        return self._guess_best_accessor_id_parts(
            self._get_filename_ext_to_format_openers(),
            self._get_format_to_data_type_aliases_openers(),
            data_id,
            data_type,
            require,
        )

    def _guess_best_writer_id_parts(
        self, data_id: str, data_type: DataTypeLike = None, require=True
    ) -> Optional[tuple[str, str, str]]:
        return self._guess_best_accessor_id_parts(
            self._get_filename_ext_to_format_writers(),
            self._get_format_to_data_type_aliases_writers(),
            data_id,
            data_type,
            require,
        )

    def _guess_all_accessor_id_parts(
        self,
        filename_ext_to_format,
        format_to_data_type_alias,
        data_id: str,
        data_type: DataTypeLike = None,
        require=True,
    ) -> list[tuple[str, str, str]]:
        assert_given(data_id, "data_id")
        ext = self._get_filename_ext(data_id)
        if data_type:
            data_type_aliases = [DataType.normalize(data_type).alias]
        else:
            data_type_aliases = None
        format_id = filename_ext_to_format.get(ext.lower())
        if format_id is not None and data_type_aliases is None:
            data_type_aliases = format_to_data_type_alias.get(format_id)
        if data_type_aliases is None or format_id is None:
            if require:
                raise DataStoreError(
                    f"Cannot determine data type for data resource {data_id!r}"
                )
            return []
        return [(dta, format_id, self.protocol) for dta in data_type_aliases]

    def _guess_best_accessor_id_parts(
        self,
        filename_ext_to_format,
        format_to_data_type_alias,
        data_id: str,
        data_type: DataTypeLike = None,
        require=True,
    ) -> tuple[str, str, str]:
        accessor_id_parts = self._guess_all_accessor_id_parts(
            filename_ext_to_format,
            format_to_data_type_alias,
            data_id,
            data_type,
            require,
        )
        return accessor_id_parts[0] if len(accessor_id_parts) > 0 else accessor_id_parts

    def _get_filename_ext(self, data_path: str) -> str:
        dot_pos = data_path.rfind(".")
        if dot_pos == -1:
            return ""
        slash_pos = data_path.rfind("/")
        if dot_pos < slash_pos:
            return ""
        # TODO: avoid self.fs here to ensure laziness.
        if self.fs.sep != "/":
            sep_pos = data_path.rfind(self.fs.sep)
            if dot_pos < sep_pos:
                return ""
        return data_path[dot_pos:]

    def _generate_data_ids(
        self,
        dir_path: str,
        data_type: DataType,
        return_tuples: bool,
        current_depth: int,
    ) -> _DataIds:
        root = self.root + ("/" + dir_path if dir_path else "")
        if not self.fs.exists(root):
            return
        # noinspection PyArgumentEqualDefault
        for file_info in self.fs.ls(root, detail=True):
            file_path: str = file_info["name"]
            if file_path.startswith(self.root):
                file_path = file_path[len(self.root) + 1 :]
            elif file_path.startswith("/" + self.root):
                file_path = file_path[len(self.root) + 2 :]
            if not file_path:
                continue
            if self._is_data_type_available(file_path, data_type):
                yield (file_path, {}) if return_tuples else file_path
            elif file_info.get("type") == "directory" and (
                self._max_depth is None or current_depth < self._max_depth
            ):
                yield from self._generate_data_ids(
                    file_path, data_type, return_tuples, current_depth + 1
                )

    def _filter_data_ids(self, data_ids: _DataIds, return_tuples: bool) -> _DataIds:
        for data_id in data_ids:
            path = data_id[0] if return_tuples else data_id
            excluded = False
            for pattern in self._excludes:
                excluded = fnmatch.fnmatch(path, pattern)
                if excluded:
                    break
            if not excluded:
                included = True
                for pattern in self._includes:
                    included = fnmatch.fnmatch(path, pattern)
                    if included:
                        break
                if included:
                    yield data_id

    @staticmethod
    def _normalize_wc(wc: Optional[Union[str, Sequence[str]]]) -> tuple[str]:
        return tuple() if not wc else (wc,) if isinstance(wc, str) else tuple(wc)


class FsDataStore(BaseFsDataStore, FsAccessor):
    """Specialization of a :class:`BaseFsDataStore` that
    also implements a :class:`FsAccessor` which serves
    the filesystem.

    The data store is capable of filtering the data identifiers reported
    by ``get_data_ids()``. For this purpose the optional keywords
    `excludes` and `includes` are used which can both take the form of
    a wildcard pattern or a sequence of wildcard patterns:

    * ``excludes``: if given and if any pattern matches the identifier,
      the identifier is not reported.
    * ``includes``: if not given or if any pattern matches the identifier,
      the identifier is reported.

    Args:
        root: Root or base directory. Defaults to "".
        max_depth: Maximum recursion depth. None means limitless.
            Defaults to 1.
        read_only: Whether this is a read-only store. Defaults to False.
        includes: Optional sequence of wildcards that include certain
            filesystem paths. Affects the data identifiers (paths)
            returned by `get_data_ids()`. By default, all paths are
            included.
        excludes: Optional sequence of wildcards that exclude certain
            filesystem paths. Affects the data identifiers (paths)
            returned by `get_data_ids()`. By default, no paths are
            excluded.
        storage_options: Parameters specific to the underlying
            filesystem. Used to instantiate the filesystem.
    """

    def __init__(
        self,
        root: str = "",
        max_depth: Optional[int] = 1,
        read_only: bool = False,
        includes: Optional[Sequence[str]] = None,
        excludes: Optional[Sequence[str]] = None,
        storage_options: dict[str, Any] = None,
    ):
        self._storage_options = storage_options or {}
        super().__init__(
            root=root,
            max_depth=max_depth,
            read_only=read_only,
            includes=includes,
            excludes=excludes,
        )

    @property
    def protocol(self) -> str:
        # Base class returns self.fs.protocol which
        # instantiates the filesystem.
        # Avoid this, as we know the protocol up front.
        return self.get_protocol()

    @property
    def storage_options(self) -> dict[str, Any]:
        return self._storage_options

    def _load_fs(self) -> fsspec.AbstractFileSystem:
        # Note, this is invoked only once per store instance.
        fs, _, _ = self.load_fs({STORAGE_OPTIONS_PARAM_NAME: self._storage_options})
        return fs

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        return cls.add_storage_options_to_params_schema(
            super().get_data_store_params_schema()
        )

    @classmethod
    def get_data_types(cls) -> tuple[str, ...]:
        _, format_to_data_type_aliases = cls._set_infos_from_extensions(
            find_data_opener_extensions, cls.get_protocol()
        )
        return tuple(
            {
                data_type
                for types_tuple in format_to_data_type_aliases.values()
                for data_type in types_tuple
            }
        )

    @classmethod
    def get_filename_extensions(
        cls, accessor_type: Literal["openers", "writers"] = "openers"
    ) -> Mapping[str, list[str]]:
        """Returns a mapping from filename extensions to lists of
        data accessor ids that open data from this format.

        The method either returns mappings to data opener ids
        or to data writer ids, depending on the setting of accessor type.

        Args:
            accessor_type: Either "openers" or "writers",
                indicates whether mappings of file extensions
                to data openers or writers shall be returned.
                Default is "openers"

        Returns:
            A mapping from filename extensions to lists of data accessor ids
        """
        if accessor_type == "openers":
            find_extensions = find_data_opener_extensions
        elif accessor_type == "writers":
            find_extensions = find_data_writer_extensions
        else:
            raise DataStoreError(
                f"Invalid accessor type. "
                f"Must be 'openers' or 'writers', was '{accessor_type}'"
            )
        filename_extensions = {}
        filename_ext_to_format, format_to_data_type_aliases = (
            cls._set_infos_from_extensions(find_extensions, cls.get_protocol())
        )
        for filename_ext, fmt in filename_ext_to_format.items():
            data_type_aliases = format_to_data_type_aliases.get(fmt, [])
            for data_type_alias in data_type_aliases:
                predicate = get_data_accessor_predicate(
                    data_type_alias, fmt, cls.get_protocol()
                )
                extensions = find_extensions(predicate)
                ext_names = [ext.name for ext in extensions]
                if filename_ext in filename_extensions:
                    filename_extensions[filename_ext] += ext_names
                else:
                    filename_extensions[filename_ext] = ext_names
        return filename_extensions
