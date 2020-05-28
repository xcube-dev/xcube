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
from typing import Mapping, Sequence, Union

from xcube.core.store.param import ParamDescriptor
from xcube.core.store.registry import Registry
from xcube.core.store.search import DatasetSearch
from xcube.core.store.search import DatasetSearchResult
from xcube.core.store.store import CubeStore

class CubeService:
    def __init__(self,
                 service_id: str,
                 cube_store: CubeStore,
                 description: str = None):
        self.id = service_id
        self.cube_store = cube_store
        self.description = description

    def search_datasets(self,
                        dataset_search: DatasetSearch) -> DatasetSearchResult:
        return self.cube_store.search_datasets(self, dataset_search)


class CubeServiceRegistry(Registry[CubeService]):
    pass
