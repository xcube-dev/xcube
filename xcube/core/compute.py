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

import warnings
from typing import Tuple, Sequence, Dict, Any, Callable, Union

import numpy as np
import xarray as xr

from xcube.core.schema import CubeSchema
from xcube.core.store import CubeStore
from xcube.core.verify import assert_cube

CubeFuncOutput = Union[xr.DataArray, np.ndarray, Sequence[Union[xr.DataArray, np.ndarray]]]
CubeFunc = Callable[..., CubeFuncOutput]


# TODO: support vectorize = all cubes have same variables and cube_func receives variables as vectors (with extra dim)

def compute_cube(cube_func: CubeFunc,
                 *input_cubes: xr.Dataset,
                 input_cube_schema: CubeSchema = None,
                 input_var_names: Sequence[str] = None,
                 input_params: Dict[str, Any] = None,
                 output_var_name: str = 'output',
                 output_var_dtype: Any = np.float64,
                 output_var_attrs: Dict[str, Any] = None,
                 vectorize: bool = False,
                 cube_asserted: bool = False) -> xr.Dataset:
    """
    Compute a new output data cube with a single variable named *output_var_name*
    from variables named *input_var_names* contained in zero, one, or more
    input data cubes in *input_cubes* using a cube factory function *cube_func*.

    *cube_func* is called concurrently for each of the chunks of the input variables.

    If *input_cubes* is not empty, *cube_func* receives variables as specified by *input_var_names*.
    If *input_cubes* is empty, *input_var_names* must be empty too, and *input_cube_schema*
    must be given, so that a new cube can be created.

    The expected signature of *cube_func* is:::

        def cube_func(*input_vars: np.ndarray,
                      input_params: Dict[str, Any],
                      dim_coords: Dict[str, np.ndarray],
                      dim_ranges: Dict[str, Tuple[int, int]]) -> np.ndarray:
            pass

    Where ``input_vars`` are the variables according to the given *input_var_names*,
    ``input_params`` is this call's *input_params*, and ``dim_ranges`` contains the current
    chunk's index ranges for each dimension.

    :param cube_func: The cube factory function.
    :param input_cubes: An optional sequence of input cube datasets, must be provided if *input_cube_schema* is not.
    :param input_cube_schema: An optional input cube schema, must be provided if *input_cubes* is not.
    :param input_var_names: A sequence of variable names
    :param input_params: Optional dictionary with processing parameters passed to *cube_func*.
    :param output_var_name: Optional name of the output variable, defaults to ``'output'``.
    :param output_var_dtype: Optional numpy datatype of the output variable, defaults to ``'float32'``.
    :param output_var_attrs: Optional metadata attributes for the output variable.
    :param vectorize: Whether all *input_cubes* have the same variables which are concatenated and passed as vectors
        to *cube_func*. Not implemented yet.
    :param cube_asserted: If False, *cube* will be verified, otherwise it is expected to be a valid cube.
    :return: A new dataset that contains the computed output variable.
    """
    if vectorize:
        raise NotImplementedError()

    if not cube_asserted:
        for cube in input_cubes:
            assert_cube(cube)

    if input_cubes:
        input_cube_schema = CubeSchema.new(input_cubes[0])
        for cube in input_cubes:
            if not cube_asserted:
                assert_cube(cube)
            if cube != input_cubes[0]:
                # noinspection PyUnusedLocal
                other_schema = CubeSchema.new(cube)
                # TODO (forman): broadcast all cubes to same shape, rechunk to same chunks
    elif input_cube_schema is None:
        raise ValueError('input_cube_schema must be given')

    if output_var_name is None:
        output_var_name = 'output'

    input_var_names = input_var_names or []
    input_vars = []
    for var_name in input_var_names:
        var = None
        for cube in input_cubes:
            if var_name in cube.data_vars:
                var = cube[var_name]
                break
        if var is None:
            raise ValueError(f'variable {var_name!r} not found in any of cubes')
        input_vars.append(var)

    def cube_func_wrapper(index_chunk, *input_var_chunks):
        nonlocal input_cube_schema, input_var_names, input_params, input_vars

        index_chunk = index_chunk.ravel()

        if index_chunk.size < 2 * input_cube_schema.ndim:
            warnings.warn(f"weird index_chunk of size {index_chunk.size} received!")
            return

        dim_ranges = {}
        for i in range(input_cube_schema.ndim):
            dim_name = input_cube_schema.dims[i]
            start = int(index_chunk[2 * i + 0])
            end = int(index_chunk[2 * i + 1])
            dim_ranges[dim_name] = start, end

        dim_coords = {}
        for coord_var_name, coord_var in input_cube_schema.coords.items():
            coord_slices = [slice(None)] * coord_var.ndim
            for i in range(input_cube_schema.ndim):
                dim_name = input_cube_schema.dims[i]
                if dim_name in coord_var.dims:
                    j = coord_var.dims.index(dim_name)
                    coord_slices[j] = slice(*dim_ranges[dim_name])
            dim_coords[coord_var_name] = coord_var[tuple(coord_slices)].values

        return cube_func(*input_var_chunks,
                         input_params=input_params,
                         dim_coords=dim_coords,
                         dim_ranges=dim_ranges)

    index_var = _gen_index_var(input_cube_schema)

    output_var = xr.apply_ufunc(cube_func_wrapper,
                                index_var,
                                *input_vars,
                                dask='parallelized',
                                output_dtypes=[output_var_dtype])
    if output_var_attrs:
        output_var.attrs.update(output_var_attrs)
    return xr.Dataset({output_var_name: output_var}, coords=input_cube_schema.coords)


def _gen_index_var(cube_schema: CubeSchema):
    dims = cube_schema.dims
    shape = cube_schema.shape
    chunks = cube_schema.chunks

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

    dataset = xr.open_zarr(store)
    index_var = dataset.__index_var__
    index_var = index_var.assign_coords(**cube_schema.coords)
    return index_var
