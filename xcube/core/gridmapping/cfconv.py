# The MIT License (MIT)
# Copyright (c) 2023 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Optional, Union, Sequence, Tuple

import pyproj
import pyproj.exceptions
import xarray as xr

from xcube.util.assertions import assert_instance
from xcube.util.assertions import assert_true
from .base import CRS_WGS84
from .base import GridMapping
from .coords import new_grid_mapping_from_coords


def find_grid_mapping_for_var(dataset: xr.Dataset,
                              var_name: str) -> Optional[GridMapping]:
    assert_instance(dataset, xr.Dataset, "dataset")
    assert_instance(var_name, str, "var_name")
    assert_true(var_name in dataset,
                message=f"variable {var_name!r} not found in dataset")

    var = dataset[var_name]

    gm_value = var.attrs.get("grid_mapping")
    if isinstance(gm_value, str) and gm_value:
        crs, gm_name, xy_coords = _find_grid_mapping_for_var_with_gm_value(
            dataset, var, gm_value
        )
    else:
        crs, gm_name, xy_coords = CRS_WGS84, "latitude_longitude", None

    if xy_coords is None:
        xy_coords = _find_coordinates_for_crs_and_gm_name(
            dataset, var, gm_name
        )

    return new_grid_mapping_from_coords(x_coords=xy_coords[0],
                                        y_coords=xy_coords[1],
                                        crs=crs)


def _find_grid_mapping_for_var_with_gm_value(
        dataset: xr.Dataset,
        var: xr.DataArray,
        gm_value: str
) -> Tuple[pyproj.CRS,
           str,
           Optional[Tuple[xr.DataArray, xr.DataArray]]]:
    xy_coords = None

    if ":" in gm_value:
        # gm_value has format "<gm_var_name>: <x_name> <y_name>"
        gm_var_name, coord_var_names = gm_value.split(":", maxsplit=1)
        coord_var_names = coord_var_names.split()
        if len(coord_var_names) == 2:
            x_name, y_name = coord_var_names
            # Check CF, whether the following is correct:
            # if gm_name in ("latitude_longitude",
            #                "rotated_latitude_longitude"):
            #     x_name, y_name = y_name, x_name
            x_coords = dataset.get(x_name)
            y_coords = dataset.get(y_name)
            if ((_is_valid_1d_coord_var(var, x_coords)
                 and _is_valid_1d_coord_var(var, y_coords))
                    or (_is_valid_2d_coord_var(var, x_coords)
                        and _is_valid_2d_coord_var(var, y_coords))):
                xy_coords = x_coords, y_coords
        if xy_coords is None:
            raise ValueError(f"invalid coordinates in"
                             f" grid mapping value {gm_value!r}")
    else:
        # gm_value has format "<gm_var_name>"
        gm_var_name = gm_value

    crs, gm_name = _parse_crs(dataset, gm_var_name)

    return crs, gm_name, xy_coords


def _find_coordinates_for_crs_and_gm_name(
        dataset: xr.Dataset,
        var: xr.DataArray,
        gm_name: str
) -> Optional[Tuple[xr.DataArray, xr.DataArray]]:
    other_vars = [dataset[var_name]
                  for var_name in dataset.variables.keys()
                  if var_name != var.name]

    valid_1d_coord_vars = [
        other_var for other_var in other_vars
        if _is_valid_1d_coord_var(var, other_var)
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
        return x_coords, y_coords

    valid_2d_coord_vars = [
        other_var for other_var in other_vars
        if _is_valid_2d_coord_var(var, other_var)
    ]
    x_coords = _find_coord_var_by_standard_name(valid_2d_coord_vars, x_name)
    y_coords = _find_coord_var_by_standard_name(valid_2d_coord_vars, y_name)
    if x_coords is not None and y_coords is not None \
            and x_coords.dims == y_coords.dims:
        return x_coords, y_coords

    xy_coords = _find_1d_coord_var_by_common_names(
        valid_1d_coord_vars,
        (("lon", "lat"),
         ("longitude", "latitude"),
         ("x", "y"),
         ("xc", "yc")),
    )
    if xy_coords is not None:
        return xy_coords

    # Check: also try _find_2d_coord_var_by_common_names()?

    raise ValueError(f"cannot determine grid mapping"
                     f" coordinates for variable {var.name!r}")


def _find_1d_coord_var_by_common_names(
        coords: Sequence[xr.DataArray],
        common_xy_names: Sequence[Tuple[str, str]],
) -> Optional[Tuple[xr.DataArray, xr.DataArray]]:
    x_coords = None
    y_coords = None
    for var in coords:
        if var.attrs.get("axis") == "X":
            x_coords = var
        if var.attrs.get("axis") == "Y":
            y_coords = var
    if x_coords is not None and y_coords is not None:
        return x_coords, y_coords

    x_coords = None
    y_coords = None
    for var in coords:
        for x_name, y_name in common_xy_names:
            if var.dims[0] == x_name and var.name == x_name:
                x_coords = var
            if var.dims[0] == y_name and var.name == y_name:
                y_coords = var
    if x_coords is not None and y_coords is not None:
        return x_coords, y_coords

    x_coords = None
    y_coords = None
    for var in coords:
        for x_name, y_name in common_xy_names:
            if var.dims[0] == x_name:
                x_coords = var
            if var.dims[0] == y_name:
                y_coords = var
    if x_coords is not None and y_coords is not None:
        return x_coords, y_coords

    return None


def _find_coord_var_by_standard_name(
        coords: Sequence[xr.DataArray],
        standard_name: str
) -> Optional[xr.DataArray]:
    for var in coords:
        if var.attrs.get("standard_name") == standard_name:
            return var
    return None


def _parse_crs(dataset: xr.Dataset,
               gm_var_name: str) -> Tuple[pyproj.CRS, Optional[str]]:
    if gm_var_name not in dataset:
        raise ValueError(f"grid mapping variable {gm_var_name!r}"
                         f" not found in dataset")
    gm_var = dataset[gm_var_name]
    try:
        return (pyproj.CRS.from_cf(gm_var.attrs),
                gm_var.attrs.get("grid_mapping_name"))
    except pyproj.exceptions.CRSError as e:
        raise ValueError(f"illegal value for"
                         f" grid mapping variable {gm_var_name!r}") from e


def _is_valid_1d_coord_var(data_var: xr.DataArray,
                           coord_var: Optional[xr.DataArray]) -> bool:
    return (coord_var is not None
            and coord_var.ndim == 1
            and coord_var.dims[0] in data_var.dims)


def _is_valid_2d_coord_var(data_var: xr.DataArray,
                           coord_var: Optional[xr.DataArray]) -> bool:
    return (coord_var is not None
            and coord_var.ndim == 2
            and coord_var.dims[0] in data_var.dims
            and coord_var.dims[1] in data_var.dims)



