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

from typing import Union, Mapping, Tuple, Optional

import xarray as xr

from .processor import DatasetTransformer
from ...gridmapping import GridMapping


class CubeRechunker(DatasetTransformer):
    """Force cube to have chunks compatible with Zarr."""

    def __init__(self, chunks: Mapping[str, Union[None, int]]):
        self._chunks = dict(chunks)

    def transform_dataset(self, cube: xr.Dataset, gm: GridMapping) \
            -> Tuple[xr.Dataset, GridMapping]:
        dim_chunks = self._chunks

        chunked_cube = xr.Dataset(attrs=cube.attrs)

        # Coordinate variables SHOULD NOT BE chunked
        chunked_cube = chunked_cube.assign_coords(
            coords={var_name: var.chunk({d: None for d in var.dims})
                    for var_name, var in cube.coords.items()}
        )

        # Data variables SHALL BE chunked according
        # to dim sizes in dim_chunks
        chunked_cube = chunked_cube.assign(
            variables={
                var_name: var.chunk({
                    var.dims[axis]: dim_chunks.get(
                        var.dims[axis],
                        _default_chunk_size(var.chunks, axis)
                    )
                    for axis in range(var.ndim)
                })
                for var_name, var in cube.data_vars.items()
            }
        )

        # Update chunks encoding for Zarr
        for var in chunked_cube.variables.values():
            if var.chunks is not None:
                var.encoding.update(chunks=[sizes[0]
                                            for sizes in var.chunks])

        return chunked_cube, gm


def _default_chunk_size(chunks: Optional[Tuple[Tuple[int, ...], ...]],
                        axis: int) -> Union[str, int]:
    if chunks is not None:
        sizes = chunks[axis]
        num_blocks = len(sizes)
        if num_blocks > 1:
            return max(*sizes)
        if num_blocks == 1:
            return sizes[0]
    return 'auto'
