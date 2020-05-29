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

import abc
from typing import Any, Dict, Optional, ItemsView

import xarray as xr

from xcube.core.store.dataset import DatasetDescriptor
from xcube.core.store.param import ParamDescriptorSet
from xcube.core.store.param import ParamValues
from xcube.core.store.search import DatasetSearch
from xcube.core.store.search import DatasetSearchResult
from xcube.core.store.service import CubeService


# TODO: list dataset_ids
# TODO: list datasets for dataset_id
# TODO: search for variables
# TODO: maybe change design using mixins CubeFinder, CubeOpener, CubeWriter, CubeTimeSliceUpdater
#       then we can define new stores in a flexible manner:
#           MyCubeStore(CubeStore, CubeFinder, CubeOpener)
#           MyCubeStore(CubeStore, CubeWriter)
#           MyCubeStore(CubeStore, CubeFinder, CubeOpener, CubeWriter, CubeTimeSliceUpdater)


class CubeStore(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def id(self) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_cube_service_params(self) -> ParamDescriptorSet:
        """Get descriptors of parameters that must or can be used to instantiate a new service object."""
        raise NotImplementedError()

    @abc.abstractmethod
    def new_cube_service(self, service_params: ParamValues = None) -> CubeService:
        raise NotImplementedError()

    @abc.abstractmethod
    def search_datasets(self,
                        cube_service: CubeService,
                        dataset_search: DatasetSearch) -> DatasetSearchResult:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_open_cube_params(self, dataset_id: str) -> ParamDescriptorSet:
        """Get descriptors of parameters that must or can be used to open a cube."""
        raise NotImplementedError()

    @abc.abstractmethod
    def open_cube(self,
                  cube_service: CubeService,
                  dataset_id: str,
                  open_params: ParamValues = None,
                  cube_params: ParamValues = None) -> xr.Dataset:
        raise NotImplementedError()


class WritableCubeStore(CubeStore, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_write_cube_params(self) -> ParamDescriptorSet:
        """Get descriptors of parameters that must or can be used to write a cube."""
        raise NotImplementedError()

    @abc.abstractmethod
    def write_cube(self,
                   cube_service: CubeService,
                   dataset: xr.Dataset,
                   replace: bool = False,
                   write_params: ParamValues = None) -> DatasetDescriptor:
        raise NotImplementedError()

    @abc.abstractmethod
    def delete_cube(self,
                    cube_service: CubeService,
                    dataset_id: str) -> DatasetDescriptor:
        raise NotImplementedError()

    @abc.abstractmethod
    def insert_cube_time_slice(self,
                               cube_service: CubeService,
                               dataset_id: str,
                               time_slice: xr.Dataset,
                               time_index: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def append_cube_time_slice(self,
                               cube_service: CubeService,
                               dataset_id: str,
                               time_slice: xr.Dataset):
        raise NotImplementedError()

    @abc.abstractmethod
    def replace_cube_time_slice(self,
                                cube_service: CubeService,
                                dataset_id: str,
                                time_slice: xr.Dataset,
                                time_index: int):
        raise NotImplementedError()


class CubeStoreRegistry:
    _DEFAULT = None

    def __init__(self):
        self._cube_stores = dict()

    def get(self, cube_store_id: str) -> Optional[CubeStore]:
        return self._cube_stores.get(cube_store_id)

    def put(self, cube_store_id: str, cube_store: CubeStore):
        self._cube_stores[cube_store_id] = cube_store

    def items(self) -> ItemsView[str, CubeStore]:
        return self._cube_stores.items()

    @classmethod
    def default(cls):
        if cls._DEFAULT is None:
            cls._DEFAULT = cls()
        return cls._DEFAULT

