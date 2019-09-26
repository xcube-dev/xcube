from typing import Tuple, Sequence, Dict, Any, Callable, Union

import numpy as np
import xarray as xr

from xcube.util.schema import CubeSchema
from .verify import assert_cube
from ..util.cubestore import CubeStore

CubeFunc = Callable[..., np.ndarray]


# TODO: support vectorize = all cubes have same variables and cube_func receives variables as vectors (with extra dim)
# TODO: support multiple outputs = provide output_variables, a mapping from variable names to dtype and other attributes

def apply(cube_func: CubeFunc,
          *cubes: xr.Dataset,
          var_names: Sequence[str] = None,
          params: Dict[str, Any] = None,
          schema: CubeSchema = None,
          vectorize: bool = False,
          output_var_defs: Sequence[Union[str, Dict[str, Any]]] = None,
          cube_asserted: bool = False):

    if not cube_asserted:
        for cube in cubes:
            assert_cube(cube)

    if cubes:
        schema = CubeSchema.new(cubes[0])
        for cube in cubes:
            if not cube_asserted:
                assert_cube(cube)
            if cube != cubes[0]:
                other_schema = CubeSchema.new(cube)
                # TODO (forman): broadcast all cubes to same shape, rechunk to same chunks
    elif schema is None:
        raise ValueError('schema must be given')

    if vectorize:
        raise NotImplementedError()

    if output_var_defs is None:
        output_var_defs = ['result']

    var_names = var_names or []
    variables = []
    for var_name in var_names:
        var = None
        for cube in cubes:
            if var_name in cube.data_vars:
                var = cube[var_name]
                break
        if var is None:
            raise ValueError(f'variable {var_name!r} not found in any of cubes')
        variables.append(var)

    index_var = gen_index_var(schema)

    def cube_func_wrapper(index_chunk, *var_chunks, **kwargs):
        nonlocal schema, var_names, variables

        index_chunk = index_chunk.ravel()
        dim_slices = {}
        for i in range(schema.ndim):
            dim_name = schema.dims[i]
            start = int(index_chunk[2 * i + 0])
            end = int(index_chunk[2 * i + 1])
            dim_slices[dim_name] = slice(start, end)

        sub_coords = {}
        for coord_var_name, coord_var in schema.coords.items():
            coord_slice_indexes = [slice(None)] * coord_var.ndim
            for i in range(schema.ndim):
                dim_name = schema.dims[i]
                j = coord_var.dims.index(dim_name)
                coord_slice_indexes[j] = dim_slices[dim_name]
            sub_coords[coord_var_name] = coord_var[tuple(coord_slice_indexes)]

        sub_vars = {}
        for i in range(len(var_names)):
            sub_var_coords = {name: sub_coords[name] for name in variables[i].coords}
            sub_var = xr.DataArray(var_chunks[i],
                                   name=var_name,
                                   coords=sub_var_coords,
                                   attrs=variables[i].attrs)
            sub_vars[var_name] = sub_var

        sub_cube = xr.Dataset(sub_vars, coords=sub_coords)

        return cube_func(sub_cube,
                         parameters=kwargs['parameters'],
                         dim_slices=dim_slices)

    # noinspection PyTypeChecker
    output_var_names = [v if isinstance(v, str) else v['name']
                        for v in output_var_defs]
    # noinspection PyTypeChecker
    output_dtypes = [np.float64 if isinstance(v, str) or 'dtype' not in v else v['dtype']
                     for v in output_var_defs]
    # noinspection PyTypeChecker,PyUnresolvedReferences
    output_attrs = [{} if isinstance(v, str) else {ak: av
                                                   for ak, av in v.items()
                                                   if ak not in {'name', 'dtype'}}
                    for v in output_var_defs]

    output_vars = xr.apply_ufunc(cube_func_wrapper,
                                 index_var,
                                 *variables,
                                 dask='parallelized',
                                 output_dtypes=output_dtypes,
                                 kwargs=params)

    if isinstance(output_vars, xr.DataArray):
        output_vars = (output_vars,)

    output_data_vars = {}
    for i in range(len(output_var_defs)):
        output_data_vars[output_var_names[i]] = xr.DataArray(output_vars[i],
                                                             attrs=output_attrs[i])
    return xr.Dataset(output_data_vars, coords=schema.coords)


def gen_index_var(schema: CubeSchema):
    dims = schema.dims
    shape = schema.shape
    chunks = schema.chunks

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
    index_var.assign_coords(**schema.coords)
    return index_var
