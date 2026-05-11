# Copyright (c) 2018-2026 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import inspect
import warnings
from collections.abc import Sequence
from typing import AbstractSet, Any, Callable, Dict, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr

from xcube.core.schema import CubeSchema
from xcube.core.verify import assert_cube

CubeFuncOutput = Union[
    xr.DataArray, np.ndarray, Sequence[Union[xr.DataArray, np.ndarray]]
]
CubeFunc = Callable[..., CubeFuncOutput]

_PREDEFINED_KEYWORDS = ["input_params", "dim_coords", "dim_ranges"]


# TODO: support vectorize = all cubes have same variables and cube_func receives variables as vectors (with extra dim)


def compute_cube(
    cube_func: CubeFunc,
    *input_cubes: xr.Dataset,
    input_cube_schema: CubeSchema = None,
    input_var_names: Sequence[str] = None,
    input_params: dict[str, Any] = None,
    output_var_name: str = "output",
    output_var_dtype: Any = np.float64,
    output_var_attrs: dict[str, Any] = None,
    vectorize: bool = None,
    cube_asserted: bool = False,
) -> xr.Dataset:
    """Compute a new output data cube with a single variable named
    *output_var_name* from variables named *input_var_names* contained in
    zero, one, or more input data cubes in *input_cubes* using a cube
    factory function *cube_func*.

    For a more detailed description of the function usage,
    please refer to :func:`compute_dataset`.

    Args:
        cube_func: The cube factory function.
        input_cubes: An optional sequence of input cube datasets,
            must be provided if *input_cube_schema* is not.
        input_cube_schema: An optional input cube schema,
            must be provided if *input_cubes* is not.
            Will be ignored if *input_cubes* is provided.
        input_var_names: A sequence of variable names
        input_params: Optional dictionary with processing parameters
            passed to *cube_func*.
        output_var_name: Optional name of the output variable,
            defaults to ``'output'``.
        output_var_dtype: Optional numpy datatype of the output variable,
            defaults to ``'float32'``.
        output_var_attrs: Optional metadata attributes for the output variable.
        vectorize: Whether all *input_cubes* have the same variables which
            are concatenated and passed as vectors
            to *cube_func*. Not implemented yet.
        cube_asserted: If False, *cube* will be verified,
            otherwise it is expected to be a valid cube.
    Returns:
        A new dataset that contains the computed output variable.
    """
    return compute_dataset(
        cube_func,
        *input_cubes,
        input_cube_schema=input_cube_schema,
        input_var_names=input_var_names,
        input_params=input_params,
        output_var_name=output_var_name,
        output_var_dtype=output_var_dtype,
        output_var_attrs=output_var_attrs,
        vectorize=vectorize,
        cube_asserted=cube_asserted,
    )


def compute_dataset(
    cube_func: CubeFunc,
    *input_cubes: xr.Dataset,
    input_cube_schema: CubeSchema = None,
    input_var_names: Sequence[str] = None,
    input_params: dict[str, Any] = None,
    output_var_name: str = "output",
    output_var_dims: AbstractSet[str] = None,
    output_var_dtype: Any = np.float64,
    output_var_attrs: dict[str, Any] = None,
    vectorize: bool = None,
    cube_asserted: bool = False,
) -> xr.Dataset:
    """Compute a new output dataset with a single variable named *output_var_name*
    from variables named *input_var_names* contained in zero, one, or more
    input data cubes in *input_cubes* using a cube factory function *cube_func*.

    *cube_func* is called concurrently for each of the chunks of the input variables.
    It is expected to return a chunk block whith is type ``np.ndarray``.

    If *input_cubes* is not empty, *cube_func* receives variables as specified by *input_var_names*.
    If *input_cubes* is empty, *input_var_names* must be empty too, and *input_cube_schema*
    must be given, so that a new cube can be created.

    The full signature of *cube_func* is:::

        def cube_func(*input_vars: np.ndarray,
                      input_params: Dict[str, Any] = None,
                      dim_coords: Dict[str, np.ndarray] = None,
                      dim_ranges: Dict[str, Tuple[int, int]] = None) -> np.ndarray:
            pass

    The arguments are:

    * ``input_vars``: the variables according to the given *input_var_names*;
    * ``input_params``: is this call's *input_params*, a mapping from parameter name to value;
    * ``dim_coords``: a mapping from dimension names to the current chunk's coordinate arrays;
    * ``dim_ranges``: a mapping from dimension names to the current chunk's index ranges.

    Only the ``input_vars`` argument is mandatory. The keyword arguments
    ``input_params``, ``input_params``, ``input_params`` do need to be present at all.

    *output_var_dims* my be given in the case, where ...
    TODO: describe new output_var_dims...

    Args:
        cube_func: The cube factory function.
        *input_cubes: An optional sequence of input cube datasets, must
            be provided if *input_cube_schema* is not.
        input_cube_schema: An optional input cube schema, must be
            provided if *input_cubes* is not.
        input_var_names: A sequence of variable names
        input_params: Optional dictionary with processing parameters
            passed to *cube_func*.
        output_var_name: Optional name of the output variable, defaults
            to ``'output'``.
        output_var_dims: Optional set of names of the output dimensions,
            used in the case *cube_func* reduces dimensions.
        output_var_dtype: Optional numpy datatype of the output
            variable, defaults to ``'float32'``.
        output_var_attrs: Optional metadata attributes for the output
            variable.
        vectorize: Whether all *input_cubes* have the same variables
            which are concatenated and passed as vectors to *cube_func*.
            Not implemented yet.
        cube_asserted: If False, *cube* will be verified, otherwise it
            is expected to be a valid cube.

    Returns:
        A new dataset that contains the computed output variable.
    """
    if vectorize is not None:
        # TODO: support vectorize = all cubes have same variables and cube_func
        #       receives variables as vectors (with extra dim)
        raise NotImplementedError("vectorize is not supported yet")

    if not cube_asserted:
        for cube in input_cubes:
            assert_cube(cube)

    # Check compatibility of inputs
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
        raise ValueError("input_cube_schema must be given")

    output_var_name = output_var_name or "output"

    # Collect named input variables, raise if not found
    input_var_names = input_var_names or []
    input_vars = []
    for var_name in input_var_names:
        input_var = None
        for cube in input_cubes:
            if var_name in cube.data_vars:
                input_var = cube[var_name]
                break
        if input_var is None:
            raise ValueError(f"variable {var_name!r} not found in any of cubes")
        input_vars.append(input_var)

    # Find out, if cube_func uses any of _PREDEFINED_KEYWORDS
    has_input_params, has_dim_coords, has_dim_ranges = _inspect_cube_func(
        cube_func, input_var_names
    )

    def cube_func_wrapper(index_chunk, *input_var_chunks):
        nonlocal input_cube_schema, input_var_names, input_params, input_vars
        nonlocal has_input_params, has_dim_coords, has_dim_ranges

        # Note, xarray.apply_ufunc does a test call with empty input arrays,
        # so index_chunk.size == 0 is a valid case
        empty_call = index_chunk.size == 0

        # TODO: when output_var_dims is given, index_chunk must be reordered
        #   as core dimensions are moved to the and of index_chunk and input_var_chunks
        if not empty_call:
            index_chunk = index_chunk.ravel()

        if index_chunk.size < 2 * input_cube_schema.ndim:
            if not empty_call:
                warnings.warn(
                    f"unexpected index_chunk of size {index_chunk.size} received!"
                )
                return None

        dim_ranges = None
        if has_dim_ranges or has_dim_coords:
            dim_ranges = {}
            for i in range(input_cube_schema.ndim):
                dim_name = input_cube_schema.dims[i]
                if not empty_call:
                    start = int(index_chunk[2 * i + 0])
                    end = int(index_chunk[2 * i + 1])
                    dim_ranges[dim_name] = start, end
                else:
                    dim_ranges[dim_name] = ()

        dim_coords = None
        if has_dim_coords:
            dim_coords = {}
            for coord_var_name, coord_var in input_cube_schema.coords.items():
                coord_slices = [slice(None)] * coord_var.ndim
                for i in range(input_cube_schema.ndim):
                    dim_name = input_cube_schema.dims[i]
                    if dim_name in coord_var.dims:
                        j = coord_var.dims.index(dim_name)
                        coord_slices[j] = slice(*dim_ranges[dim_name])
                dim_coords[coord_var_name] = coord_var[tuple(coord_slices)].values

        kwargs = {}
        if has_input_params:
            kwargs["input_params"] = input_params
        if has_dim_ranges:
            kwargs["dim_ranges"] = dim_ranges
        if has_dim_coords:
            kwargs["dim_coords"] = dim_coords

        return cube_func(*input_var_chunks, **kwargs)

    index_var = _gen_index_var(input_cube_schema)

    all_input_vars = [index_var] + input_vars

    input_core_dims = None
    if output_var_dims:
        input_core_dims = []
        has_warned = False
        for i in range(len(all_input_vars)):
            input_var = all_input_vars[i]
            var_core_dims = [
                dim for dim in input_var.dims if dim not in output_var_dims
            ]
            must_rechunk = False
            if var_core_dims and input_var.chunks:
                for var_core_dim in var_core_dims:
                    dim_index = input_var.dims.index(var_core_dim)
                    dim_chunk_size = input_var.chunks[dim_index][0]
                    dim_shape_size = input_var.shape[dim_index]
                    if dim_chunk_size != dim_shape_size:
                        must_rechunk = True
                        break
            if must_rechunk:
                if not has_warned:
                    warnings.warn(
                        f"Input variables must not be chunked in dimension(s): {', '.join(var_core_dims)}.\n"
                        f"Rechunking applies, which may drastically decrease runtime performance "
                        f"and increase memory usage."
                    )
                    has_warned = True
                all_input_vars[i] = input_var.chunk(
                    {var_core_dim: -1 for var_core_dim in var_core_dims}
                )
            input_core_dims.append(var_core_dims)

    output_var = xr.apply_ufunc(
        cube_func_wrapper,
        *all_input_vars,
        dask="parallelized",
        input_core_dims=input_core_dims,
        output_dtypes=[output_var_dtype],
    )
    if output_var_attrs:
        output_var.attrs.update(output_var_attrs)
    return xr.Dataset({output_var_name: output_var}, coords=input_cube_schema.coords)


def _inspect_cube_func(cube_func: CubeFunc, input_var_names: Sequence[str] = None):
    (
        args,
        varargs,
        varkw,
        defaults,
        kwonlyargs,
        kwonlydefaults,
        annotations,
    ) = inspect.getfullargspec(cube_func)
    cube_func_name = "?"
    if hasattr(cube_func, "__name__"):
        cube_func_name = cube_func.__name__
    true_args = [arg not in _PREDEFINED_KEYWORDS for arg in args]
    if False in true_args and any(true_args[true_args.index(False) :]):
        raise ValueError(
            f"invalid cube_func {cube_func_name!r}: "
            f"any argument must occur before any of {', '.join(_PREDEFINED_KEYWORDS)}, "
            f"but got {', '.join(args)}"
        )
    if not all(true_args) and varargs:
        raise ValueError(
            f"invalid cube_func {cube_func_name!r}: "
            f"any argument must occur before any of {', '.join(_PREDEFINED_KEYWORDS)}, "
            f"but got {', '.join(args)} before *{varargs}"
        )
    num_input_vars = len(input_var_names) if input_var_names else 0
    num_args = sum(true_args)
    if varargs is None and num_input_vars != num_args:
        raise ValueError(
            f"invalid cube_func {cube_func_name!r}: "
            f"expected {num_input_vars} arguments, "
            f"but got {', '.join(args)}"
        )
    has_input_params = "input_params" in args or "input_params" in kwonlyargs
    has_dim_coords = "dim_coords" in args or "dim_coords" in kwonlyargs
    has_dim_ranges = "dim_ranges" in args or "dim_ranges" in kwonlyargs
    return has_input_params, has_dim_coords, has_dim_ranges



def _gen_index_var(cube_schema: CubeSchema):
    dims = cube_schema.dims
    shape = cube_schema.shape
    chunks = cube_schema.chunks
    ndim = len(dims)

    def make_chunk(block, block_info=None):
        """Create one chunk filled with chunk index metadata."""
        info = block_info[None]
        chunk_shape = info["chunk-shape"]
        chunk_location = info["chunk-location"]
        data = np.zeros(chunk_shape, dtype=np.uint64)
        flat = data.ravel()

        if flat.size < ndim * 2:
            raise ValueError("chunk size too small")

        for i in range(ndim):
            start = chunks[i][0] * chunk_location[i]
            end = start + chunk_shape[i]
            flat[2 * i] = start
            flat[2 * i + 1] = end

        return data

    template = da.empty(shape, chunks=chunks, dtype=np.uint64)
    index_data = da.map_blocks(
        make_chunk,
        template,
        dtype=np.uint64,
    )
    index_var = xr.DataArray(
        index_data,
        dims=dims,
        coords=cube_schema.coords,
        name="__index_var__",
    )

    return index_var
