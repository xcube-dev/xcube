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

from typing import Union, Mapping

import xarray as xr

from .processor import CubeProcessor


class CubeRechunker(CubeProcessor):

    def __init__(self, chunks: Mapping[str, Union[None, int]]):
        self._chunks = dict(chunks)

    def process_cube(self, cube: xr.Dataset) -> xr.Dataset:
        dim_chunks = self._chunks

        chunked_cube = xr.Dataset(attrs=cube.attrs)

        # Coordinate variables WILL NOT BE chunked
        chunked_cube = chunked_cube.assign_coords(
            coords={var_name: var.chunk({d: None for d in var.dims})
                    for var_name, var in cube.coords.items()}
        )
        # Data variables WILL BE chunked according to dim sizes in dim_chunks
        chunked_cube = chunked_cube.assign(
            variables={var_name: var.chunk({d: dim_chunks.get(str(d), 'auto')
                                            for d in var.dims})
                       for var_name, var in cube.data_vars.items()}
        )
        # Update variable encoding for Zarr
        for var in chunked_cube.variables.values():
            assert var.chunks is not None, "var.chunks is not None"
            var_chunks = var.chunks
            zarr_chunks = var.ndim * [0]
            for i in range(var.ndim):
                sizes = var_chunks[i]
                if len(sizes) == 1:
                    zarr_chunks[i] = sizes[0]
                elif len(sizes) > 1:
                    zarr_chunks[i] = max(*sizes)
            var.encoding.update(chunks=zarr_chunks)

        return chunked_cube
