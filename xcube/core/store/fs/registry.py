# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
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

from typing import Type, Dict, Optional, Any

import fsspec

from .accessor import FsAccessor
from .accessor import FsDataAccessor
from .impl.cog import MultiLevelDatasetGeoTiffFsDataAccessor
from .impl.dataset import DatasetNetcdfFsDataAccessor, \
    DatasetGeoTiffFsDataAccessor
from .impl.dataset import DatasetZarrFsDataAccessor
from .impl.fs import FileFsAccessor
from .impl.fs import MemoryFsAccessor
from .impl.fs import S3FsAccessor
from .impl.geodataframe import GeoDataFrameGeoJsonFsDataAccessor
from .impl.geodataframe import GeoDataFrameShapefileFsDataAccessor
from .impl.mldataset import MultiLevelDatasetLevelsFsDataAccessor
from .store import FsDataStore
from ..assertions import assert_valid_params
from ..error import DataStoreError

############################################
# FsAccessor

_FS_ACCESSOR_CLASSES: Dict[str, Type[FsAccessor]] = {}


def register_fs_accessor_class(
        fs_accessor_class: Type[FsAccessor]
):
    """
    Register a concrete filesystem accessor class.

    :param fs_accessor_class: a concrete class
        that extends :class:FsAccessor.
    """
    protocol = fs_accessor_class.get_protocol()
    _FS_ACCESSOR_CLASSES[protocol] = fs_accessor_class


for cls in (FileFsAccessor, S3FsAccessor, MemoryFsAccessor):
    register_fs_accessor_class(cls)


def get_fs_accessor_class(protocol: str) -> Type[FsAccessor]:
    """
    Get the class for a filesystem accessor.

    :param protocol: The filesystem protocol,
        for example "file", "s3", "memory".
    :return: A class that derives from :class:FsAccessor
    """
    fs_accessor_class = _FS_ACCESSOR_CLASSES.get(protocol)
    if fs_accessor_class is None:
        try:
            fsspec.get_filesystem_class(protocol)
        except ImportError as e:
            raise DataStoreError(
                f'Filesystem for protocol {protocol!r}'
                f' is not installed or requires additional packages'
            ) from e
        except ValueError as e:
            raise DataStoreError(
                f'Filesystem not found for protocol {protocol!r}'
            ) from e

        class FsAccessorClass(FsAccessor):
            @classmethod
            def get_protocol(cls) -> str:
                return protocol

        fs_accessor_class = FsAccessorClass
    return fs_accessor_class


############################################
# FsDataAccessor

_FS_DATA_ACCESSOR_CLASSES: Dict[str, Type[FsDataAccessor]] = {}


def register_fs_data_accessor_class(
        fs_data_accessor_class: Type[FsDataAccessor]
):
    """
    Register an abstract filesystem data accessor class.

    Such data accessor classes are used to dynamically
    construct concrete data store classes by combining
    them with a concrete :class:FsAccessor.

    :param fs_data_accessor_class: an abstract class
        that extends :class:FsDataAccessor.
    """
    format_id = fs_data_accessor_class.get_format_id()
    for data_type in fs_data_accessor_class.get_data_types():
        key = f'{data_type}:{format_id}'
        _FS_DATA_ACCESSOR_CLASSES[key] = fs_data_accessor_class


for cls in (DatasetZarrFsDataAccessor,
            DatasetNetcdfFsDataAccessor,
            MultiLevelDatasetLevelsFsDataAccessor,
            GeoDataFrameShapefileFsDataAccessor,
            GeoDataFrameGeoJsonFsDataAccessor,
            DatasetGeoTiffFsDataAccessor,
            MultiLevelDatasetGeoTiffFsDataAccessor):
    register_fs_data_accessor_class(cls)


def get_fs_data_accessor_class(protocol: str,
                               data_type_alias: str,
                               format_id: str) -> Type[FsDataAccessor]:
    """
    Get the class for a filesystem data accessor.

    :param protocol: The filesystem protocol,
        for example "file", "s3", "memory".
    :param data_type_alias: The data type alias name,
        for example "dataset", "geodataframe".
    :param format_id: The format identifier,
        for example "zarr", "geojson".
    :return: A class that derives from :class:FsAccessor
    """
    accessor_id = f'{data_type_alias}:{format_id}'
    data_accessor_class = _FS_DATA_ACCESSOR_CLASSES.get(accessor_id)
    if data_accessor_class is None:
        raise DataStoreError(f'Combination of data type {data_type_alias!r}'
                             f' and format {format_id!r} is not supported')

    fs_accessor_class = get_fs_accessor_class(protocol)

    class FsDataAccessorClass(fs_accessor_class, data_accessor_class):
        pass

    # Should we set set __name_ and __doc__ properties here?

    return FsDataAccessorClass


############################################
# FsDataStore

def get_fs_data_store_class(protocol: str) -> Type[FsDataStore]:
    """
    Get the class for of a filesystem-based data store.

    :param protocol: The filesystem protocol,
        for example "file", "s3", "memory".
    :return: A class that derives from :class:FsDataStore
    """
    fs_accessor_class = get_fs_accessor_class(protocol)

    class FsDataStoreClass(fs_accessor_class, FsDataStore):
        pass

    # Should we set set __name_ and __doc__ properties here?

    return FsDataStoreClass


def new_fs_data_store(
        protocol: str,
        root: str = '',
        max_depth: Optional[int] = 1,
        read_only: bool = False,
        storage_options: Dict[str, Any] = None
) -> FsDataStore:
    """
    Create a new instance of a filesystem-based data store.

    :param protocol: The filesystem protocol,
        for example "file", "s3", "memory".
    :param root: Root or base directory.
        Defaults to "".
    :param max_depth: Maximum recursion depth. None means limitless.
        Defaults to 1.
    :param read_only: Whether this is a read-only store.
        Defaults to False.
    :param storage_options: Options specific to the underlying filesystem
        identified by *protocol*.
        Used to instantiate the filesystem.
    :return: A new data store instance of type :class:FsDataStore.
    """
    fs_data_store_class = get_fs_data_store_class(protocol)
    store_params_schema = fs_data_store_class.get_data_store_params_schema()
    store_params = {k: v
                    for k, v in dict(root=root,
                                     max_depth=max_depth,
                                     read_only=read_only,
                                     storage_options=storage_options).items()
                    if v is not None}
    assert_valid_params(store_params,
                        name='store_params',
                        schema=store_params_schema)
    return fs_data_store_class(**store_params)
