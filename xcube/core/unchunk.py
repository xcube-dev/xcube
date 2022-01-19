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

import json
import os.path
from typing import List, Sequence

import numpy as np
import xarray as xr
import zarr


def unchunk_dataset(dataset_path: str, var_names: Sequence[str] = None, coords_only: bool = False):
    """
    Unchunk dataset variables in-place.

    :param dataset_path: Path to ZARR dataset directory.
    :param var_names: Optional list of variable names.
    :param coords_only: Un-chunk coordinate variables only.
    """

    is_zarr = os.path.isfile(os.path.join(dataset_path, '.zgroup'))
    if not is_zarr:
        raise ValueError(f'{dataset_path!r} is not a valid Zarr directory')

    with xr.open_zarr(dataset_path) as dataset:
        if var_names is None:
            if coords_only:
                var_names = list(dataset.coords)
            else:
                var_names = list(dataset.variables)
        else:
            for var_name in var_names:
                if coords_only:
                    if var_name not in dataset.coords:
                        raise ValueError(f'variable {var_name!r} is not a coordinate variable in {dataset_path!r}')
                else:
                    if var_name not in dataset.variables:
                        raise ValueError(f'variable {var_name!r} is not a variable in {dataset_path!r}')

    _unchunk_vars(dataset_path, var_names)


def _unchunk_vars(dataset_path: str, var_names: List[str]):
    for var_name in var_names:
        var_path = os.path.join(dataset_path, var_name)

        # Optimization: if "shape" and "chunks" are equal in ${var}/.zarray, we are done
        var_array_info_path = os.path.join(var_path, '.zarray')
        with open(var_array_info_path, 'r') as fp:
            var_array_info = json.load(fp)
            if var_array_info.get('shape') == var_array_info.get('chunks'):
                continue

        # Open array and remove chunks from the data
        var_array = zarr.convenience.open_array(var_path, 'r+')
        if var_array.shape != var_array.chunks:
            # TODO (forman): Fully loading data is inefficient and dangerous for large arrays.
            #                Instead save unchunked to temp and replace existing chunked array dir with temp.
            # Fully load data and attrs so we no longer depend on files
            data = np.array(var_array)
            attributes = var_array.attrs.asdict()
            # Save array data
            zarr.convenience.save_array(var_path, data, chunks=False, fill_value=var_array.fill_value)
            # zarr.convenience.save_array() does not seem save user attributes (file ".zattrs" not written),
            # therefore we must modify attrs explicitly:
            var_array = zarr.convenience.open_array(var_path, 'r+')
            var_array.attrs.update(attributes)

    zarr.consolidate_metadata(dataset_path)
