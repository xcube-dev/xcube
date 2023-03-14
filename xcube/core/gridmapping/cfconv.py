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

import functools
from typing import Optional, Union, Sequence

import pyproj
import pyproj.exceptions
import xarray as xr

from xcube.util.assertions import assert_instance
from xcube.util.assertions import assert_true
from .base import GridMapping
from .coords import new_grid_mapping_from_coords


def find_grid_mapping_for_var(dataset: xr.Dataset,
                              var_name: str,
                              strict: bool = False) -> Optional[GridMapping]:
    assert_instance(dataset, xr.Dataset, "dataset")
    assert_instance(var_name, str, "var_name")
    assert_true(var_name in dataset,
                message=f"variable {var_name!r} not found in dataset")

    var = dataset[var_name]

    gm_var_name = var.attrs.get("grid_mapping")
    if isinstance(gm_var_name, str) and gm_var_name:
        return _find_grid_mapping_for_var_with_gm_attr(
            dataset,
            var,
            gm_var_name,
            strict=strict
        )

    return None


def _find_grid_mapping_for_var_with_gm_attr(dataset: xr.Dataset,
                                            var: xr.DataArray,
                                            gm_var_name: str,
                                            strict: bool = False) \
        -> Optional[GridMapping]:
    maybe_fail = functools.partial(_maybe_fail, strict)

    if ":" in gm_var_name:
        gm_var_name, coord_var_names = gm_var_name.split(":", maxsplit=1)
        coord_var_names = coord_var_names.split()
    else:
        coord_var_names = dataset.coords.keys()

    if gm_var_name not in dataset:
        return maybe_fail(f"grid mapping variable {gm_var_name!r}"
                          f" not found in dataset")

    gm_var = dataset[gm_var_name]
    try:
        crs = pyproj.CRS.from_cf(gm_var.attrs)
    except pyproj.exceptions.CRSError as e:
        return maybe_fail(e)

    gm_name = gm_var.attrs.get("grid_mapping_name")

    coord_vars = [dataset[coord_var_name]
                  for coord_var_name in coord_var_names
                  if coord_var_name in dataset]

    if len(coord_vars) < len(coord_var_names):
        return maybe_fail(f"not all coordinate variables"
                          f" of {coord_var_names!r}"
                          f" found in dataset")

    valid_1d_coord_vars = [
        coord_var for coord_var in coord_vars
        if (coord_var is not None
            and coord_var.ndim == 1
            and coord_var.dims[0] in var.dims)
    ]

    if gm_name == "latitude_longitude":
        x_name, y_name = "longitude", "latitude"
    elif gm_name == "rotated_latitude_longitude":
        x_name, y_name = "grid_longitude", "grid_latitude"
    else:
        x_name, y_name = "projection_x_coordinate", "projection_y_coordinate"

    x_coords = _find_coord_var_by_standard_name(valid_1d_coord_vars, x_name)
    y_coords = _find_coord_var_by_standard_name(valid_1d_coord_vars, y_name)
    if x_coords is not None and y_coords is not None:
        return new_grid_mapping_from_coords(x_coords=x_coords,
                                            y_coords=y_coords,
                                            crs=crs)

    valid_2d_coord_vars = [
        coord_var for coord_var in coord_vars
        if (coord_var is not None
            and coord_var.ndim == 2
            and coord_var.dims[0] in var.dims
            and coord_var.dims[1] in var.dims)
    ]
    x_coords = _find_coord_var_by_standard_name(valid_2d_coord_vars, x_name)
    y_coords = _find_coord_var_by_standard_name(valid_2d_coord_vars, y_name)
    if x_coords is not None and y_coords is not None \
            and x_coords.dims == y_coords.dims:
        return new_grid_mapping_from_coords(x_coords=x_coords,
                                            y_coords=y_coords,
                                            crs=crs)

    return maybe_fail(f"cannot find coordinates in {coord_var_names!r}"
                      f" for grid mapping variable {gm_var_name!r}")


def _find_coord_var_by_standard_name(
        coords: Sequence[xr.DataArray],
        standard_name: str
) -> Optional[xr.DataArray]:
    return next(
        (var for var in coords
         if var.attrs.get("standard_name") == standard_name),
        None
    )


def _maybe_fail(strict: bool, error: Union[str, Exception]) -> None:
    if strict:
        if isinstance(error, str):
            raise ValueError(error)
        else:
            raise error
    return None
