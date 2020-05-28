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
from typing import Sequence

import xarray as xr

from xcube.core.store.dataset import DatasetDescriptor
from xcube.core.store.param import ParamDescriptor
from xcube.core.store.registry import Registry
from xcube.core.store.search import DatasetSearch, DatasetSearchResult
from xcube.core.store.service import CubeService


class CubeStore(metaclass=abc.ABCMeta):
    def __init__(self, store_id: str, description: str = None):
        self.id = store_id
        self.description = description

    @property
    @abc.abstractmethod
    def new_cube_service_params(self) -> Sequence[ParamDescriptor]:
        """Get descriptors of parameters that must or can be used to instantiate a new service object."""
        raise NotImplementedError()

    @abc.abstractmethod
    def new_cube_service(self, **params) -> CubeService:
        raise NotImplementedError()

    @abc.abstractmethod
    def search_datasets(self,
                        cube_service: CubeService,
                        dataset_search: DatasetSearch) -> DatasetSearchResult:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def open_cube_params(self) -> Sequence[ParamDescriptor]:
        """Get descriptors of parameters that must or can be used to open a cube."""
        raise NotImplementedError()

    @abc.abstractmethod
    def open_cube(self,
                  cube_service: CubeService,
                  dataset_id: str,
                  **params) -> xr.Dataset:
        raise NotImplementedError()

    @abc.abstractmethod
    def add_cube(self,
                 cube_service: CubeService,
                 dataset: xr.Dataset,
                 **params) -> DatasetDescriptor:
        raise NotImplementedError()

    @abc.abstractmethod
    def replace_cube(self,
                     cube_service: CubeService,
                     dataset_id: str,
                     dataset: xr.Dataset) -> DatasetDescriptor:
        raise NotImplementedError()

    @abc.abstractmethod
    def remove_cube(self,
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


class CubeStoreRegistry(Registry[CubeStore]):
    pass
