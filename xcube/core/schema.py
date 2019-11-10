# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
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

from typing import Tuple, Sequence, Dict, Optional, Mapping

import numpy as np
import xarray as xr


class CubeSchema:
    """
    A schema that can be used to create new xcube datasets.
    The given *shape*, *dims*, and *chunks*, *coords* apply to all data variables.

    :param shape: A tuple of dimension sizes.
    :param coords: A dictionary of coordinate variables. Must have values for all *dims*.
    :param dims: A sequence of dimension names. Defaults to ``('time', 'lat', 'lon')``.
    :param chunks: A tuple of chunk sizes in each dimension.
    """

    def __init__(self,
                 shape: Sequence[int],
                 coords: Mapping[str, np.array],
                 dims: Sequence[str] = None,
                 chunks: Sequence[int] = None):

        if not shape:
            raise ValueError('shape must be a sequence of integer sizes')
        if not coords:
            raise ValueError('coords must be a mapping from dimension names to label arrays')

        ndim = len(shape)
        if ndim < 3:
            raise ValueError('shape must have at least three dimensions')
        dims = tuple(dims) or ('time', 'lat', 'lon')
        if dims and len(dims) != ndim:
            raise ValueError('dims must have same length as shape')
        if dims[0] != 'time':
            raise ValueError("the first name in dims must be 'time'")
        if dims[-2:] not in (('lat', 'lon'), ('y', 'x')):
            raise ValueError("the last two names in dims must be either ('lat', 'lon') or ('y', 'x')")
        if chunks and len(chunks) != ndim:
            raise ValueError('chunks must have same length as shape')
        for i in range(ndim):
            dim_name = dims[i]
            dim_size = shape[i]
            if dim_name not in coords:
                raise ValueError(f'missing dimension {dim_name!r} in coords')
            dim_labels = coords[dim_name]
            if len(dim_labels.shape) != 1:
                raise ValueError(f'labels of {dim_name!r} in coords must be one-dimensional')
            if len(dim_labels) != dim_size:
                raise ValueError(f'number of labels of {dim_name!r} in coords does not match shape')

        self._shape = tuple(shape)
        self._dims = dims
        self._chunks = tuple(chunks) if chunks else None
        self._coords = dict(coords)

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._dims)

    @property
    def dims(self) -> Tuple[str, ...]:
        """Tuple of dimension names."""
        return self._dims

    @property
    def time_dim(self) -> str:
        """Name of time dimension."""
        return self._dims[0]

    @property
    def spatial_dims(self) -> Tuple[str, str]:
        """Tuple (pair) of spatial dimension names, will be either ('lat', 'lon') or ('y', 'x')."""
        return self._dims[-2:]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Tuple of dimension sizes."""
        return self._shape

    @property
    def chunks(self) -> Optional[Tuple[int]]:
        """Tuple of dimension chunk sizes."""
        return self._chunks

    @property
    def coords(self) -> Dict[str, xr.DataArray]:
        """Dictionary of coordinate variables."""
        return self._coords

    @classmethod
    def new(cls, cube: xr.Dataset) -> "CubeSchema":
        """Create a cube schema from given *cube*."""
        return get_cube_schema(cube)

    def _repr_html_(self):
        """Return a HTML representation for Jupyter Notebooks."""
        return (
            f'<table>'
            f'<tr><td>Shape:</td><td>{self.shape}</td></tr>'
            f'<tr><td>Chunk sizes:</td><td>{self.chunks}</td></tr>'
            f'<tr><td>Dimensions:</td><td>{self.dims}</td></tr>'
            f'</table>'
        )


# TODO (forman): code duplication with xcube.core.verify._check_data_variables(), line 76
def get_cube_schema(cube: xr.Dataset) -> CubeSchema:
    """
    Derive cube schema from given *cube*.

    :param cube: The data cube.
    :return: The cube schema.
    """
    first_dims = None
    first_shape = None
    first_chunks = None
    first_coords = None

    for var_name, var in cube.data_vars.items():

        dims = var.dims
        if first_dims is None:
            first_dims = dims
        elif first_dims != dims:
            raise ValueError(f'all variables must have same dimensions, but variable {var_name!r} '
                             f'has dimensions {dims!r}')

        shape = var.shape
        if first_shape is None:
            first_shape = shape
        elif first_shape != shape:
            raise ValueError(f'all variables must have same shape, but variable {var_name!r} '
                             f'has shape {shape!r}')

        coords = var.coords
        if first_coords is None:
            first_coords = coords

        dask_chunks = var.chunks
        if dask_chunks:
            chunks = []
            for i in range(var.ndim):
                dim_name = var.dims[i]
                dim_chunk_sizes = dask_chunks[i]
                first_size = dim_chunk_sizes[0]
                if any(size != first_size for size in dim_chunk_sizes[1:-1]):
                    raise ValueError(f'dimension {dim_name!r} of variable {var_name!r} has chunks of different sizes: '
                                     f'{dim_chunk_sizes!r}')
                chunks.append(first_size)
            chunks = tuple(chunks)
            if first_chunks is None:
                first_chunks = chunks
            elif first_chunks != chunks:
                raise ValueError(f'all variables must have same chunks, but variable {var_name!r} '
                                 f'has chunks {chunks!r}')

    if first_dims is None:
        raise ValueError('cube is empty')

    return CubeSchema(first_shape, first_coords, dims=tuple(str(d) for d in first_dims), chunks=first_chunks)
