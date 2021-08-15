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
from typing import Type, Dict, TypeVar

import fsspec

from xcube.core.store import DataStoreError
from .common import FileFsAccessor
from .common import FsAccessor
from .common import FsDataAccessor
from .common import MemoryFsAccessor
from .common import S3FsAccessor
from .dataset import DatasetNetcdfFsDataAccessor
from .dataset import DatasetZarrFsDataAccessor
from .geodataframe import GeoDataFrameGeoJsonFsDataAccessor
from .geodataframe import GeoDataFrameShapefileFsDataAccessor
from .store import FsDataStore

FSA = TypeVar('FSA', bound=FsAccessor)
_FS_ACCESSOR_CLASSES: Dict[str, FsAccessor] = {}


############################################
# FileSystemAccessor


def register_fs_accessor_class(
        fs_accessor_class: Type[FSA]
):
    """
    Register a concrete file system accessor class.
    :param fs_accessor_class: a concrete class
        that extends :class:FsAccessor.
    """
    fs_protocol = fs_accessor_class.get_fs_protocol()
    _FS_ACCESSOR_CLASSES[fs_protocol] = fs_accessor_class


for cls in (FileFsAccessor, S3FsAccessor, MemoryFsAccessor):
    register_fs_accessor_class(cls)


def get_fs_accessor_class(storage_id):
    fs_protocol = storage_id
    fs_accessor_class = _FS_ACCESSOR_CLASSES.get(fs_protocol)
    if fs_accessor_class is None:
        try:
            fsspec.get_filesystem_class(fs_protocol)
        except (ValueError, ImportError) as e:
            raise DataStoreError(
                f'unknown file system protocol {fs_protocol!r}'
            ) from e

        class FileSystemAccessorClass(FsAccessor):
            @classmethod
            def get_fs_protocol(cls) -> str:
                return fs_protocol

        fs_accessor_class = FileSystemAccessorClass
    return fs_accessor_class


############################################
# FileSystemDataAccessor

FSDA = TypeVar('FSDA', bound=FsDataAccessor)
_FS_DATA_ACCESSOR_CLASSES: Dict[str, FsDataAccessor] = {}


def register_fs_data_accessor_class(
        fs_data_accessor_class: Type[FSDA]
):
    """
    Register an abstract file system data accessor class.

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


def get_fs_data_accessor_class(type_specifier: str,
                               format_id: str,
                               storage_id: str) -> Type:
    accessor_id = f'{type_specifier}:{format_id}'
    data_accessor_class = _FS_DATA_ACCESSOR_CLASSES.get(accessor_id)
    if data_accessor_class is None:
        raise DataStoreError(f'Combination of data type {type_specifier!r}'
                             f' and format {format_id!r} is not supported')

    fs_accessor_class = get_fs_accessor_class(storage_id)

    class FsDataAccessorClass(fs_accessor_class, data_accessor_class):
        pass

    return FsDataAccessorClass


############################################
# FileSystemDataStore

def get_fs_data_store_class(storage_id: str) -> Type:
    fs_accessor_class = get_fs_accessor_class(storage_id)

    class FsDataStoreClass(fs_accessor_class, FsDataStore):
        pass

    return FsDataStoreClass
