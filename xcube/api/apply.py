from typing import Tuple, Sequence, Dict, Any, Callable

import numpy as np
import xarray as xr

from .verify import assert_cube
from ..util.cubestore import CubeStore

CubeFunc = Callable[..., np.ndarray]


def apply(cube: xr.Dataset,
          cube_func: CubeFunc,
          data_var_names: Sequence[str] = None,
          coord_var_names: Sequence[str] = None,
          parameters: Dict[str, Any] = None,
          cube_asserted: bool = False):
    if not cube_asserted:
        assert_cube(cube)

    dims, shape, chunks = get_cube_schema(cube)
    ndim = len(dims)

    index_var = gen_index_var(dims, shape, chunks)
    index_var = index_var.assign_coords(**{var_name: var for var_name, var in cube.coords.items() if var_name in dims})

    def cube_func_wrapper(index_chunk, *var_chunks, **kwargs):
        nonlocal ndim, dims

        coord_vars = kwargs['coord_vars']
        index_chunk = index_chunk.ravel()
        coord_chunks = []
        for i in range(ndim):
            dim_name = dims[i]
            if dim_name in coord_vars:
                coord_var = coord_vars[dim_name]
                start = int(index_chunk[2 * i + 0])
                end = int(index_chunk[2 * i + 1])
                coord_chunks.append(coord_var[start: end].values)

        return cube_func(*var_chunks, *coord_chunks, kwargs['parameters'])

    data_var_names = data_var_names or []
    coord_var_names = coord_var_names or []

    data_vars = [cube.data_vars[var_name] for var_name in data_var_names]
    coord_vars = {var_name: cube.coords[var_name] for var_name in coord_var_names}

    return xr.apply_ufunc(cube_func_wrapper,
                          index_var,
                          *data_vars,
                          dask='parallelized',
                          output_dtypes=[np.float],
                          kwargs={'coord_vars': coord_vars, 'parameters': parameters})


def gen_index_var(dims, shape, chunks):
    # noinspection PyUnusedLocal
    def get_chunk(cube_store: CubeStore, name: str, index: Tuple[int, ...]) -> bytes:
        data = np.zeros(cube_store.chunks, dtype=np.uint64)
        data_view = data.ravel()
        if data_view.base is not data:
            raise ValueError('view expected')
        if data_view.size < cube_store.ndim * 2:
            raise ValueError('size too small')
        for i in range(cube_store.ndim):
            j1 = cube_store.chunks[i] * index[i]
            j2 = j1 + cube_store.chunks[i]
            data_view[2 * i] = j1
            data_view[2 * i + 1] = j2
        return data.tobytes()

    store = CubeStore(dims, shape, chunks)
    store.add_lazy_array('__index_var__', '<u8', get_chunk=get_chunk)

    ds = xr.open_zarr(store)
    return ds.__index_var__


# TODO (forman): code duplication with xcube.api.verify._check_data_variables(), line 76
def get_cube_schema(cube: xr.Dataset, cube_asserted: bool = False) -> Tuple[
    Tuple[str, ...], Tuple[int, ...], Tuple[int, ...]]:
    if not cube_asserted:
        assert_cube(cube)

    first_dims = None
    first_shape = None
    first_chunks = None

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

    return first_dims, first_shape, first_chunks
