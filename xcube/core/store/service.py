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
import xarray as xr

from xcube.core.store.param import ParamValues, ParamDescriptorSet
from xcube.core.store.search import DatasetSearch
from xcube.core.store.search import DatasetSearchResult
from xcube.core.store.store import CubeStore, WritableCubeStore


class CubeService:
    def __init__(self, cube_store: CubeStore, service_params: ParamValues):
        self._cube_store = cube_store
        self._service_params = service_params

    def search_datasets(self, dataset_search: DatasetSearch) -> DatasetSearchResult:
        return self._cube_store.search_datasets(self, dataset_search)

    def get_open_cube_params(self, dataset_id: str) -> ParamDescriptorSet:
        return self._cube_store.get_open_cube_params(dataset_id)

    def open_cube(self, dataset_id: str, open_params: ParamValues) -> xr.Dataset:
        return self._cube_store.open_cube(self, dataset_id, open_params)


class WritableCubeService(CubeService):
    def __init__(self, cube_store: WritableCubeStore, service_params: ParamValues):
        super().__init__(cube_store, service_params)

    def get_write_cube_params(self) -> ParamDescriptorSet:
        # noinspection PyUnresolvedReferences
        return self._cube_store.get_write_cube_params()

    def write_cube(self, cube: xr.Dataset, write_params: ParamValues):
        # noinspection PyUnresolvedReferences
        return self._cube_store.write_cube(self, cube, write_params)
