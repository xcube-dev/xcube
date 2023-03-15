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

from typing import Optional, Sequence, Tuple, Dict, Hashable, List

import pyproj
import pyproj.exceptions
import xarray as xr

from xcube.util.assertions import assert_instance
from xcube.util.assertions import assert_true
from .base import CRS_WGS84

DimPair = Tuple[Hashable, Hashable]
CoordsPair = Tuple[xr.DataArray, xr.DataArray]
GridMappingTuple = Tuple[pyproj.CRS, str, CoordsPair]


def find_grid_mappings_for_data_vars(dataset: xr.Dataset) \
        -> Dict[Hashable, GridMappingTuple]:
    grid_mappings = find_grid_mappings_for_dataset(dataset)
    vars_to_grid_mappings: Dict[Hashable, GridMappingTuple] = {}
    for var_name, var in dataset.data_vars.items():
        if var.ndim < 2:
            continue
        for gmt in grid_mappings:
            _, _, xy_coords = gmt
            x_dim, y_dim = _get_xy_dims_from_xy_coords(xy_coords)
            if x_dim in var.dims and y_dim in var.dims:
                vars_to_grid_mappings[var_name] = gmt
                break
    return vars_to_grid_mappings


def find_grid_mappings_for_dataset(dataset: xr.Dataset) \
        -> List[GridMappingTuple]:
    dims_to_grid_mappings: Dict[DimPair, GridMappingTuple] = {}
    for var_name, var in dataset.data_vars.items():
        found = False
        for x_dim, y_dim in dims_to_grid_mappings.keys():
            if x_dim in var.dims and y_dim in var.dims:
                found = True
                break
        if not found:
            gmt = find_grid_mapping_for_data_var(dataset, var_name)
            _, _, xy_coords = gmt
            xy_dims = _get_xy_dims_from_xy_coords(xy_coords)
            dims_to_grid_mappings[xy_dims] = gmt
    return list(dims_to_grid_mappings.values())


def _get_xy_dims_from_xy_coords(xy_coords: CoordsPair) -> DimPair:
    x_coords, y_coords = xy_coords
    if x_coords.ndim == 1:
        # 1-D coordinates
        assert y_coords.ndim == 1
        return x_coords.dims[0], y_coords.dims[0]
    else:
        # 2-D coordinates
        assert x_coords.ndim == 2
        assert x_coords.dims == y_coords.dims
        return x_coords.dims[1], x_coords.dims[0]


def find_grid_mapping_for_data_var(
        dataset: xr.Dataset,
        var_name: Hashable
) -> Optional[GridMappingTuple]:
    assert_instance(dataset, xr.Dataset, "dataset")
    assert_instance(var_name, Hashable, "var_name")
    assert_true(var_name in dataset,
                message=f"variable {var_name!r} not found in dataset")

    var = dataset[var_name]
    if var.ndim < 2:
        return None

    gm_value = var.attrs.get("grid_mapping")
    if isinstance(gm_value, str) and gm_value:
        crs, gm_name, xy_coords = _find_grid_mapping_for_var_with_gm_value(
            dataset, var, gm_value
        )
        force_xy_coords = True
    else:
        crs, gm_name, xy_coords = CRS_WGS84, "latitude_longitude", None
        force_xy_coords = False

    if xy_coords is None:
        xy_coords = _find_coordinates_for_crs_and_gm_name(
            dataset, var, gm_name, force_xy_coords
        )
        if xy_coords is None:
            return None

    return crs, gm_name, xy_coords


def _find_grid_mapping_for_var_with_gm_value(
        dataset: xr.Dataset,
        var: xr.DataArray,
        gm_value: str
) -> Tuple[pyproj.CRS,
           str,
           Optional[CoordsPair]]:
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
        gm_name: str,
        force: bool
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

    if force:
        raise ValueError(f"cannot determine grid mapping"
                         f" coordinates for variable {var.name!r}"
                         f" with dimensions {var.dims!r}")
    return None

def _find_1d_coord_var_by_common_names(
        coords: Sequence[xr.DataArray],
        common_xy_names: Sequence[Tuple[str, str]],
) -> Optional[Tuple[xr.DataArray, xr.DataArray]]:

    # Priority 1: find var by "axis" attribute
    x_coords = None
    y_coords = None
    for var in coords:
        if var.attrs.get("axis") == "X":
            x_coords = var
        if var.attrs.get("axis") == "Y":
            y_coords = var
    if x_coords is not None and y_coords is not None:
        return x_coords, y_coords

    # Priority 2: find var where dim name == common name == var name
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

    # Priority 3: find var where dim name == common name
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
        raise ValueError(f"variable {gm_var_name!r}"
                         f" is not a valid grid mapping") from e


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



