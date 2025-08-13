# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Optional

import fsspec

from ..accessor import find_data_opener_extensions, find_data_writer_extensions
from ..assertions import assert_valid_params
from ..error import DataStoreError
from .accessor import FsAccessor, FsDataAccessor
from .impl.dataset import DatasetNetcdfFsDataAccessor, DatasetZarrFsDataAccessor
from .impl.fs import (
    AzureFsAccessor,
    FileFsAccessor,
    FtpFsAccessor,
    HttpsFsAccessor,
    MemoryFsAccessor,
    S3FsAccessor,
)
from .impl.geodataframe import (
    GeoDataFrameGeoJsonFsDataAccessor,
    GeoDataFrameShapefileFsDataAccessor,
)
from .impl.mldataset import (
    DatasetLevelsFsDataAccessor,
    MultiLevelDatasetLevelsFsDataAccessor,
)
from .impl.rasterio import (
    DatasetGeoTiffFsDataAccessor,
    DatasetJ2kFsDataAccessor,
    MultiLevelDatasetGeoTiffFsDataAccessor,
    MultiLevelDatasetJ2kFsDataAccessor,
)
from .store import FsDataStore

############################################
# FsAccessor

_FS_ACCESSOR_CLASSES: dict[str, type[FsAccessor]] = {}


def register_fs_accessor_class(fs_accessor_class: type[FsAccessor]):
    """Register a concrete filesystem accessor class.

    Args:
        fs_accessor_class: a concrete class that extends
            :class:`FsAccessor`.
    """
    protocol = fs_accessor_class.get_protocol()
    _FS_ACCESSOR_CLASSES[protocol] = fs_accessor_class


for cls in (
    AzureFsAccessor,
    FileFsAccessor,
    FtpFsAccessor,
    HttpsFsAccessor,
    MemoryFsAccessor,
    S3FsAccessor,
):
    register_fs_accessor_class(cls)


def get_fs_accessor_class(protocol: str) -> type[FsAccessor]:
    """Get the class for a filesystem accessor.

    Args:
        protocol: The filesystem protocol, for example "file", "s3",
            "memory".

    Returns:
        A class that derives from :class:`FsAccessor`
    """
    fs_accessor_class = _FS_ACCESSOR_CLASSES.get(protocol)
    if fs_accessor_class is None:
        try:
            fsspec.get_filesystem_class(protocol)
        except ImportError as e:
            raise DataStoreError(
                f"Filesystem for protocol {protocol!r}"
                f" is not installed or requires additional packages"
            ) from e
        except ValueError as e:
            raise DataStoreError(
                f"Filesystem not found for protocol {protocol!r}"
            ) from e

        class FsAccessorClass(FsAccessor):
            @classmethod
            def get_protocol(cls) -> str:
                return protocol

        fs_accessor_class = FsAccessorClass
    return fs_accessor_class


############################################
# FsDataAccessor

_FS_DATA_ACCESSOR_CLASSES: dict[str, type[FsDataAccessor]] = {}


def register_fs_data_accessor_class(fs_data_accessor_class: type[FsDataAccessor]):
    """Register an abstract filesystem data accessor class.

    Such data accessor classes are used to dynamically
    construct concrete data store classes by combining
    them with a concrete :class:`FsAccessor`.

    Args:
        fs_data_accessor_class: an abstract class that extends
            :class:`FsDataAccessor`.
    """
    data_type = fs_data_accessor_class.get_data_type()
    format_id = fs_data_accessor_class.get_format_id()
    key = f"{data_type.alias}:{format_id}"
    _FS_DATA_ACCESSOR_CLASSES[key] = fs_data_accessor_class


for cls in (
    DatasetZarrFsDataAccessor,
    DatasetNetcdfFsDataAccessor,
    DatasetGeoTiffFsDataAccessor,
    DatasetJ2kFsDataAccessor,
    DatasetLevelsFsDataAccessor,
    MultiLevelDatasetGeoTiffFsDataAccessor,
    MultiLevelDatasetJ2kFsDataAccessor,
    MultiLevelDatasetLevelsFsDataAccessor,
    GeoDataFrameShapefileFsDataAccessor,
    GeoDataFrameGeoJsonFsDataAccessor,
):
    register_fs_data_accessor_class(cls)


def get_fs_data_accessor_class(
    protocol: str, data_type_alias: str, format_id: str
) -> type[FsDataAccessor]:
    """Get the class for a filesystem data accessor.

    Args:
        protocol: The filesystem protocol, for example "file", "s3",
            "memory".
        data_type_alias: The data type alias name, for example
            "dataset", "geodataframe".
        format_id: The format identifier, for example "zarr", "geojson".

    Returns:
        A class that derives from :class:`FsAccessor`
    """
    accessor_id = f"{data_type_alias}:{format_id}"
    data_accessor_class = _FS_DATA_ACCESSOR_CLASSES.get(accessor_id)
    if data_accessor_class is None:
        raise DataStoreError(
            f"Combination of data type {data_type_alias!r}"
            f" and format {format_id!r} is not supported"
        )

    fs_accessor_class = get_fs_accessor_class(protocol)

    class FsDataAccessorClass(fs_accessor_class, data_accessor_class):
        pass

    # Should we set __name_ and __doc__ properties here?

    return FsDataAccessorClass


############################################
# FsDataStore


def get_fs_data_store_class(protocol: str) -> type[FsDataStore]:
    """Get the class for of a filesystem-based data store.

    Args:
        protocol: The filesystem protocol, for example "file", "s3",
            "memory".

    Returns:
        A class that derives from :class:`FsDataStore`
    """
    fs_accessor_class = get_fs_accessor_class(protocol)

    class FsDataStoreClass(fs_accessor_class, FsDataStore):
        pass

    # Should we set set __name_ and __doc__ properties here?

    return FsDataStoreClass


def new_fs_data_store(
    protocol: str,
    root: str = "",
    max_depth: Optional[int] = 1,
    read_only: bool = False,
    includes: Optional[Sequence[str]] = None,
    excludes: Optional[Sequence[str]] = None,
    storage_options: dict[str, Any] = None,
) -> FsDataStore:
    """Create a new instance of a filesystem-based data store.

    The data store is capable of filtering the data identifiers reported
    by ``get_data_ids()``. For this purpose the optional keywords
    `excludes` and `includes` are used which can both take the form of
    a wildcard pattern or a sequence of wildcard patterns:

    * ``excludes``: if given and if any pattern matches the identifier,
      the identifier is not reported.
    * ``includes``: if not given or if any pattern matches the identifier,
      the identifier is reported.

    Args:
        protocol: The filesystem protocol, for example "file", "s3",
            "memory".
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
        storage_options: Options specific to the underlying filesystem
            identified by *protocol*. Used to instantiate the
            filesystem.

    Returns:
        A new data store instance of type :class:`FsDataStore`.
    """
    fs_data_store_class = get_fs_data_store_class(protocol)
    store_params_schema = fs_data_store_class.get_data_store_params_schema()
    store_params = {
        k: v
        for k, v in dict(
            root=root,
            max_depth=max_depth,
            read_only=read_only,
            includes=includes,
            excludes=excludes,
            storage_options=storage_options,
        ).items()
        if v is not None
    }
    assert_valid_params(store_params, name="store_params", schema=store_params_schema)
    return fs_data_store_class(**store_params)


def get_filename_extensions(
    accessor_type: Literal["openers", "writers"] = "openers",
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
    for ext in find_extensions():
        print(ext.name)
        file_extensions = ext.metadata.get("extensions", [])
        for file_extension in file_extensions:
            if file_extension in filename_extensions:
                filename_extensions[file_extension] += [ext.name]
            else:
                filename_extensions[file_extension] = [ext.name]
    return filename_extensions
