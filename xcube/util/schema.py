from typing import Tuple, Sequence, Dict, Optional

import numpy as np
import xarray as xr


class CubeSchema:
    def __init__(self,
                 shape: Sequence[int],
                 coords: Dict[str, np.array],
                 dims: Sequence[str] = None,
                 chunks: Sequence[int] = None):

        if not shape:
            raise ValueError('shape must a sequence of integer sizes')
        if not coords:
            raise ValueError('coords must a mapping from dimension names to label arrays')

        ndim = len(shape)
        dims = dims or ('time', 'lat', 'lon')
        if dims and len(dims) != ndim:
            raise ValueError('dims must have same length as shape')
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
        self._dims = tuple(dims)
        self._chunks = tuple(chunks) if chunks else None
        self._coords = dict(coords)

    @property
    def ndim(self) -> int:
        return len(self._dims)

    @property
    def dims(self) -> Tuple[str]:
        return self._dims

    @property
    def shape(self) -> Tuple[int]:
        return self._shape

    @property
    def chunks(self) -> Optional[Tuple[int]]:
        return self._chunks

    @property
    def coords(self) -> Dict[str, xr.DataArray]:
        return self._coords

    @classmethod
    def new(cls, cube: xr.Dataset, cube_asserted: bool = False) -> "CubeSchema":
        return get_cube_schema(cube)


# TODO (forman): code duplication with xcube.api.verify._check_data_variables(), line 76
def get_cube_schema(cube: xr.Dataset) -> CubeSchema:
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
        if dask_chunks is None:
            raise ValueError(f'all variables must be chunked, but variable {var_name!r} is not')
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

    return CubeSchema(first_shape, first_coords, dims=first_dims, chunks=first_chunks)
