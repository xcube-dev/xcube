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

from xcube.core.store import DataStoreError
from .accessor import FsAccessor
from .accessor import FsDataAccessor
from .impl.dataset import DatasetNetcdfFsDataAccessor
from .impl.dataset import DatasetZarrFsDataAccessor
from .impl.fs import FileFsAccessor
from .impl.fs import MemoryFsAccessor
from .impl.fs import S3FsAccessor
from .impl.geodataframe import GeoDataFrameGeoJsonFsDataAccessor
from .impl.geodataframe import GeoDataFrameShapefileFsDataAccessor
from .store import FsDataStore

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
    fs_protocol = fs_accessor_class.get_fs_protocol()
    _FS_ACCESSOR_CLASSES[fs_protocol] = fs_accessor_class


for cls in (FileFsAccessor, S3FsAccessor, MemoryFsAccessor):
    register_fs_accessor_class(cls)


def get_fs_accessor_class(fs_protocol: str) -> Type[FsAccessor]:
    """
    Get the class for a filesystem accessor.

    :param fs_protocol: The filesystem protocol,
        for example "file", "s3", "memory".
    :return: A class that derives from :class:FsAccessor
    """
    fs_accessor_class = _FS_ACCESSOR_CLASSES.get(fs_protocol)
    if fs_accessor_class is None:
        try:
            fsspec.get_filesystem_class(fs_protocol)
        except ImportError as e:
            raise DataStoreError(
                f'Filesystem for protocol {fs_protocol!r}'
                f' is not installed or requires additional packages'
            ) from e
        except ValueError as e:
            raise DataStoreError(
                f'Filesystem not found for protocol {fs_protocol!r}'
            ) from e

        class FsAccessorClass(FsAccessor):
            @classmethod
            def get_fs_protocol(cls) -> str:
                return fs_protocol

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
    type_specifier = fs_data_accessor_class.get_type_specifier()
    format_id = fs_data_accessor_class.get_format_id()
    key = f'{type_specifier}:{format_id}'
    _FS_DATA_ACCESSOR_CLASSES[key] = fs_data_accessor_class


for cls in (DatasetZarrFsDataAccessor,
            DatasetNetcdfFsDataAccessor,
            GeoDataFrameShapefileFsDataAccessor,
            GeoDataFrameGeoJsonFsDataAccessor):
    register_fs_data_accessor_class(cls)


def get_fs_data_accessor_class(fs_protocol: str,
                               type_specifier: str,
                               format_id: str) -> Type[FsDataAccessor]:
    """
    Get the class for a filesystem data accessor.

    :param type_specifier: The data type specifier,
        for example "dataset", "geodataframe".
    :param format_id: The format identifier,
        for example "zarr", "geojson".
    :param fs_protocol: The filesystem protocol,
        for example "file", "s3", "memory".
    :return: A class that derives from :class:FsAccessor
    """
    accessor_id = f'{type_specifier}:{format_id}'
    data_accessor_class = _FS_DATA_ACCESSOR_CLASSES.get(accessor_id)
    if data_accessor_class is None:
        raise DataStoreError(f'Combination of data type {type_specifier!r}'
                             f' and format {format_id!r} is not supported')

    fs_accessor_class = get_fs_accessor_class(fs_protocol)

    class FsDataAccessorClass(fs_accessor_class, data_accessor_class):
        pass

    # Should we set set __name_ and __doc__ properties here?

    return FsDataAccessorClass


############################################
# FsDataStore

def get_fs_data_store_class(fs_protocol: str) -> Type[FsDataStore]:
    """
    Get the class for of a filesystem-based data store.

    :param fs_protocol: The filesystem protocol,
        for example "file", "s3", "memory".
    :return: A class that derives from :class:FsDataStore
    """
    fs_accessor_class = get_fs_accessor_class(fs_protocol)

    class FsDataStoreClass(fs_accessor_class, FsDataStore):
        pass

    # Should we set set __name_ and __doc__ properties here?

    return FsDataStoreClass


def new_fs_data_store(fs_protocol: str,
                      fs_params: Dict[str, Any] = None,
                      root: str = '',
                      max_depth: Optional[int] = 1,
                      read_only: bool = False) -> FsDataStore:
    """
    Create a new instance of a filesystem-based data store.

    :param fs_protocol: The filesystem protocol,
        for example "file", "s3", "memory".
    :param fs_params: Parameters specific to the underlying filesystem
        identified by *fs_protocol*.
        Used to instantiate the filesystem.
    :param root: Root or base directory.
        Defaults to "".
    :param max_depth: Maximum recursion depth. None means limitless.
        Defaults to 1.
    :param read_only: Whether this is a read-only store.
        Defaults to False.
    :return: A new data store instance of type :class:FsDataStore.
    """
    fs_data_store_class = get_fs_data_store_class(fs_protocol)
    store_params_schema = fs_data_store_class.get_data_store_params_schema()
    store_params = {k: v
                    for k, v in dict(fs_params=fs_params,
                                     root=root,
                                     max_depth=max_depth,
                                     read_only=read_only).items()
                    if v is not None}
    store_params_schema.validate_instance(store_params)
    return fs_data_store_class(**store_params)
