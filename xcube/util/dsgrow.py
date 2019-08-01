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

import os
import os.path
import tempfile
from collections import MutableMapping
from typing import Dict, Union

import numpy as np
import xarray as xr
import zarr

from .chunk import chunk_dataset


def add_time_slice(store: Union[str, MutableMapping],
                   time_slice: xr.Dataset,
                   chunk_sizes: Dict[str, int] = None):
    insert_index = get_time_insert_index(store, time_slice)
    if insert_index == -1:
        append_time_slice(store, time_slice, chunk_sizes=chunk_sizes)
    else:
        insert_time_slice(store, insert_index, time_slice, chunk_sizes=chunk_sizes)


def get_time_insert_index(store: Union[str, MutableMapping],
                          time_slice: xr.Dataset):
    try:
        cube = xr.open_zarr(store)
    except ValueError:
        # ValueError raised if cube store does not exist
        return -1

    slice_time = time_slice.time[0]
    for i in range(cube.time.size):
        time = cube.time[i]
        if slice_time == time:
            raise NotImplementedError(f'time already found in {store}, this is not yet supported')
        if slice_time < time:
            return i
    return -1


def append_time_slice(store: Union[str, MutableMapping],
                      time_slice: xr.Dataset,
                      chunk_sizes: Dict[str, int] = None):
    if chunk_sizes:
        time_slice = chunk_dataset(time_slice, chunk_sizes, format_name='zarr')
    time_slice.to_zarr(store, mode='a', append_dim='time')
    _unchunk_zarr_cube_time_vars(store)


def insert_time_slice(store: Union[str, MutableMapping],
                      insert_index: int,
                      time_slice: xr.Dataset,
                      chunk_sizes: Dict[str, int] = None):
    time_var_names = []
    with xr.open_zarr(store) as cube:
        for var_name in cube.variables:
            var = cube[var_name]
            if var.ndim >= 1 and 'time' in var.dims:
                if var.dims[0] != 'time':
                    raise ValueError(f"Variable: {var_name} Dimension 'time' must be first dimension")
                time_var_names.append(var_name)
    if chunk_sizes:
        time_slice = chunk_dataset(time_slice, chunk_sizes, format_name='zarr')
    temp_dir = tempfile.TemporaryDirectory(suffix='time-slice-', prefix='.zarr')
    time_slice.to_zarr(temp_dir.name)
    slice_root_group = zarr.open(temp_dir.name, mode='r')
    slice_arrays = dict(slice_root_group.arrays())

    cube_root_group = zarr.open(store, mode='r+')
    for var_name, var_array in cube_root_group.arrays():
        if var_name in time_var_names:
            slice_array = slice_arrays[var_name]
            # Add one empty time step
            empty = zarr.creation.empty(slice_array.shape, dtype=var_array.dtype)
            var_array.append(empty, axis=0)
            # Shift contents
            var_array[insert_index + 1:, ...] = var_array[insert_index:-1, ...]
            # Insert slice
            var_array[insert_index, ...] = slice_array[0]

    _unchunk_zarr_cube_time_vars(store)


def _unchunk_zarr_cube_time_vars(cube_path: str):
    with xr.open_zarr(cube_path) as dataset:
        coord_var_names = [var_name for var_name in dataset.coords if 'time' in dataset[var_name].dims]
    for coord_var_name in coord_var_names:
        coord_var_path = os.path.join(cube_path, coord_var_name)
        coord_var_array = zarr.convenience.open_array(coord_var_path, 'r+')
        # Fully load data and attrs so we no longer depend on files
        data = np.array(coord_var_array)
        attributes = coord_var_array.attrs.asdict()
        # Save array data
        zarr.convenience.save_array(coord_var_path, data, chunks=False, fill_value=coord_var_array.fill_value)
        # zarr.convenience.save_array() does not seem save user attributes (file ".zattrs" not written),
        # therefore we must modify attrs explicitly:
        coord_var_array = zarr.convenience.open_array(coord_var_path, 'r+')
        coord_var_array.attrs.update(attributes)
