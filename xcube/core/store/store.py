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

from abc import ABCMeta, abstractmethod
from typing import Optional, Iterator, Mapping, Any

import xarray as xr

from xcube.core.store.dataset import DatasetDescriptor
from xcube.core.store.search import CubeSearch
from xcube.core.store.search import CubeSearchResult
from xcube.util.jsonschema import JsonObjectSchema


# TODO: IMPORTANT: support multi-resolution datasets (*.levels)
# TODO: IMPORTANT: replace, reuse, or align with
#   xcube.core.dsio.DatasetIO class
# TODO: rename to DatasetStore, DatasetFinder, DatasetOpener, DatasetWriter?

class CubeStore(metaclass=ABCMeta):
    """
    An abstract cube store.
    """

    @classmethod
    def get_cube_store_params_schema(cls) -> JsonObjectSchema:
        """
        Get descriptions of parameters that must or can be used to instantiate a new cube store object.
        Parameters are named and described by the properties of the returned JSON object schema.
        The default implementation returns JSON object schema that can have any properties.
        """
        return JsonObjectSchema(additional_properties=True)

    @abstractmethod
    def iter_cubes(self) -> Iterator[DatasetDescriptor]:
        """
        Iterate descriptors of all cubes in this store.
        :return: A cube descriptor iterator.
        """


# TODO: support search for variables
class CubeFinder(metaclass=ABCMeta):
    """
    Find cubes in this cube store.
    """

    def get_search_params_schema(self) -> JsonObjectSchema:
        """
        Get descriptions of parameters that must or can be used to search the store.
        Parameters are named and described by the properties of the returned JSON object schema.
        The default implementation returns JSON object schema that can have any properties.
        """
        return JsonObjectSchema()

    @abstractmethod
    def search_cubes(self,
                     dataset_search: CubeSearch) -> CubeSearchResult:
        """
        Searches for cubes using the given search request.

        :param dataset_search: The dataset search request.
        :return: The search result.
        """


class CubeOpener(metaclass=ABCMeta):
    """
    Open cubes in this cube store.
    """

    def get_open_cube_params_schema(self, cube_id: str) -> JsonObjectSchema:
        """
        Get descriptions of parameters that must or can be used to open a cube from the store.
        Parameters are named and described by the properties of the returned JSON object schema.
        The default implementation returns JSON object schema that can have any properties.
        """
        return JsonObjectSchema()

    @abstractmethod
    def open_cube(self,
                  cube_id: str,
                  open_params: Mapping[str, Any] = None,
                  cube_params: Mapping[str, Any] = None) -> xr.Dataset:
        """
        Open a cube from this cube store.

        :param cube_id: The cube identifier.
        :param open_params: Open parameters.
        :param cube_params: Cube generation parameters.
        :return: The cube.
        """


class CubeWriter(metaclass=ABCMeta):
    """
    Write cubes to and delete cubes from this cube store.
    """

    def get_write_cube_params_schema(self) -> JsonObjectSchema:
        """
        Get descriptions of parameters that must or can be used to write a cube to the store.
        Parameters are named and described by the properties of the returned JSON object schema.
        The default implementation returns JSON object schema that can have any properties.
        """
        return JsonObjectSchema()

    @abstractmethod
    def write_cube(self,
                   cube: xr.Dataset,
                   cube_id: str = None,
                   replace: bool = False,
                   write_params: Mapping[str, Any] = None) -> str:
        """
        Writes *cube* into the cube store and returns its cube identifier.

        :param cube: The cube to be written.
        :param cube_id: Optional cube identifier. If not given, a new one will be created.
        :param replace: Whether to replace an existing cube.
        :param write_params: Store specific writer parameters.
        :return: The cube identifier.
        :raise CubeStoreError: on error
        """

    @abstractmethod
    def delete_cube(self, cube_id: str) -> bool:
        """
        Delete the cube with the given cube identifier.

        :param cube_id: The cube identifier.
        :return: True, if the cube was deleted. False, if it does not exist.
        :raise CubeStoreError: on error
        """


class CubeTimeSliceUpdater(metaclass=ABCMeta):
    @abstractmethod
    def append_cube_time_slice(self,
                               cube_id: str,
                               time_slice: xr.Dataset):
        """
        Append a time slice to the identified cube.

        :param cube_id: The cube identifier.
        :param time_slice: The time slice to be inserted. Must be compatible with the cube.
        """

    @abstractmethod
    def insert_cube_time_slice(self,
                               cube_id: str,
                               time_slice: xr.Dataset,
                               time_index: int):
        """
        Insert a time slice into the identified cube at given index.

        :param cube_id: The cube identifier.
        :param time_slice: The time slice to be inserted. Must be compatible with the cube.
        :param time_index: The time index.
        """

    @abstractmethod
    def replace_cube_time_slice(self,
                                cube_id: str,
                                time_slice: xr.Dataset,
                                time_index: int):
        """
        Replace a time slice in the identified cube at given index.

        :param cube_id: The cube identifier.
        :param time_slice: The time slice to be inserted. Must be compatible with the cube.
        :param time_index: The time index.
        """


class CubeStoreError(Exception):
    """
    Raised on error in any of the cube store methods.

    :param message: The error message.
    :param cube_store: The cube store that caused the error.
    """

    def __init__(self, message: str, cube_store: CubeStore = None):
        super().__init__(message)
        self._cube_store = cube_store

    @property
    def cube_store(self) -> Optional[CubeStore]:
        return self._cube_store


