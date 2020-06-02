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

import os.path
import uuid
from typing import Optional, Iterator, Mapping, Any

import xarray as xr

from xcube.core.dsio import rimraf
from xcube.core.store.dataset import DatasetDescriptor
from xcube.core.store.store import CubeOpener
from xcube.core.store.store import CubeStore
from xcube.core.store.store import CubeStoreError
from xcube.core.store.store import CubeWriter
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema


# TODO: validate params
# TODO: complete tests
class DirectoryCubeStore(CubeStore, CubeOpener, CubeWriter):
    """
    A cube store that stores cubes in a directory in the local file system.

    :param base_dir: The base directory where cubes are stored.
    :param read_only: Whether this is a read-only store.
    """

    def __init__(self,
                 base_dir: str,
                 read_only: bool = False):
        self._base_dir = base_dir
        self._read_only = read_only

    @classmethod
    def get_cube_store_params_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(properties=dict(
            base_dir=JsonStringSchema(),
            read_only=JsonBooleanSchema(default=False)
        ))

    @property
    def base_dir(self) -> str:
        return self._base_dir

    @property
    def read_only(self) -> bool:
        return self._read_only

    def get_cube_path(self, cube_id: str) -> str:
        if not cube_id:
            raise CubeStoreError(f'Missing cube identifier')
        return self._base_dir

    def iter_cubes(self) -> Iterator[DatasetDescriptor]:
        for file_name in os.listdir(self._base_dir):
            file_path = os.path.join(self._base_dir, file_name)
            if os.path.isdir(file_path):
                if file_path.endswith('.zarr'):
                    base_name, _ = os.path.splitext(file_name)
                    # TODO: create DatasetDescriptor from cube
                    yield DatasetDescriptor(dataset_id=base_name)

    def get_open_cube_params_schema(self, dataset_id: str) -> JsonObjectSchema:
        return JsonObjectSchema(properties=dict(
            decode_cf=JsonBooleanSchema(default=True),
            format=JsonStringSchema(nullable=True, default=None),
        ))

    def open_cube(self,
                  cube_id: str,
                  open_params: Mapping[str, Any] = None,
                  cube_params: Mapping[str, Any] = None) -> xr.Dataset:
        cube_path = self.get_cube_path(cube_id)
        return xr.open_dataset(cube_path, **(open_params or {}))

    def get_write_cube_params_schema(self) -> JsonObjectSchema:
        return JsonObjectSchema(properties=dict(
            format=JsonStringSchema(default='zarr'),
        ))

    def write_cube(self,
                   cube: xr.Dataset,
                   cube_id: str = None,
                   replace: bool = False,
                   write_params: Mapping[str, Any] = None) -> str:
        cube_id = self._ensure_valid_cube_id(cube_id)
        cube_path = self.get_cube_path(cube_id)
        if os.path.exists(cube_path) and not replace:
            raise CubeStoreError(f'A cube named "{cube_id}" already exists', cube_store=self)
        cube.to_zarr(cube_path)
        return cube_id

    def delete_cube(self, cube_id: str) -> bool:
        cube_path = self.get_cube_path(cube_id)
        if not os.path.exists(cube_path):
            return False
        rimraf(cube_path)
        return True

    @classmethod
    def _ensure_valid_cube_id(cls, cube_id: Optional[str]) -> str:
        return cube_id or str(uuid.uuid4())
