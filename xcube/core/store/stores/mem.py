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

import uuid
import xarray as xr
from typing import Iterator, Dict, Mapping, Any, Optional

from xcube.core.store.dataset import DatasetDescriptor
from xcube.core.store.store import CubeOpener
from xcube.core.store.store import CubeStore
from xcube.core.store.store import CubeStoreError
from xcube.core.store.store import CubeWriter


class MemoryCubeStore(CubeStore, CubeOpener, CubeWriter):
    """
    An in-memory cube store.
    Its main use case is testing.
    """

    _GLOBAL_CUBE_MEMORY = dict()

    def __init__(self, cube_memory: Dict[str, Any] = None):
        self._cube_memory = cube_memory if cube_memory is not None else self.get_global_cube_memory()

    @property
    def cube_memory(self) -> Dict[str, xr.Dataset]:
        return self._cube_memory

    @classmethod
    def get_global_cube_memory(cls) -> Dict[str, xr.Dataset]:
        return cls._GLOBAL_CUBE_MEMORY

    @classmethod
    def replace_global_cube_memory(cls, global_cube_memory: Dict[str, xr.Dataset]) -> Dict[str, xr.Dataset]:
        old_global_cube_memory = cls._GLOBAL_CUBE_MEMORY
        cls._GLOBAL_CUBE_MEMORY = global_cube_memory
        return old_global_cube_memory

    def iter_cubes(self) -> Iterator[DatasetDescriptor]:
        for cube_id, cube in self._cube_memory.items():
            # TODO: create DatasetDescriptor from cube
            yield DatasetDescriptor(dataset_id=cube_id)

    def open_cube(self,
                  cube_id: str,
                  open_params: Mapping[str, Any] = None,
                  cube_params: Mapping[str, Any] = None) -> xr.Dataset:
        if cube_id not in self._cube_memory:
            raise CubeStoreError(f'Unknown cube identifier "{cube_id}"', cube_store=self)
        return self._cube_memory[cube_id]

    def write_cube(self,
                   cube: xr.Dataset,
                   cube_id: str = None,
                   replace: bool = False,
                   write_params: Mapping[str, Any] = None) -> str:
        if cube_id and cube_id in self._cube_memory and not replace:
            raise CubeStoreError(f'A cube named "{cube_id}" already exists')
        cube_id = self._ensure_valid_cube_id(cube_id)
        self._cube_memory[cube_id] = cube
        return cube_id

    def delete_cube(self, cube_id: str):
        if cube_id not in self._cube_memory:
            return False
        del self._cube_memory[cube_id]
        return True

    @classmethod
    def _ensure_valid_cube_id(cls, cube_id: Optional[str]) -> str:
        return cube_id or str(uuid.uuid4())
