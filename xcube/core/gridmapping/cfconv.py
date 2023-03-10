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

import warnings
from collections.abc import MutableMapping
from typing import Optional, Dict, Any, Hashable, Union, Set, List, Tuple

import numpy as np
import pyproj
import xarray as xr
import zarr
import zarr.convenience

from xcube.core.schema import get_dataset_chunks
from xcube.util.assertions import assert_instance
from .base import CRS_WGS84


class GridMappingSchema:
    def __init__(self,
                 name: str,
                 standard_names: Tuple[str, str],
                 common_names: Tuple[Tuple[str, str], ...],
                 default_crs: Optional[pyproj.CRS]):
        self.name = name
        self.standard_names = standard_names
        self.common_var_names = common_names
        self.default_crs = default_crs


GMS_LATITUDE_LONGITUDE = GridMappingSchema(
    'latitude_longitude',
    ('longitude', 'latitude'),
    (('lon', 'lat'),
     ('longitude', 'latitude')),
    CRS_WGS84
)

GMS_ROTATED_LATITUDE_LONGITUDE = GridMappingSchema(
    'rotated_latitude_longitude',
    ('grid_longitude', 'grid_latitude'),
    (('rlon', 'rlat'),
     ('rlongitude', 'rlatitude')),
    CRS_WGS84
)

GMS_PROJECTED = GridMappingSchema(
    'projected',
    ('projection_x_coordinate', 'projection_y_coordinate'),
    (('x', 'y'),
     ('xc', 'yc'),
     ('transformed_x', 'transformed_y')),
    None
)

GM_SCHEMAS = (GMS_LATITUDE_LONGITUDE,
              GMS_ROTATED_LATITUDE_LONGITUDE,
              GMS_PROJECTED)


class GridMappingTemplate:
    def __init__(self,
                 var_name: str,
                 var: xr.DataArray,
                 crs: pyproj.CRS,
                 x_coords: Optional[xr.DataArray] = None,
                 y_coords: Optional[xr.DataArray] = None):
        self.var_name = var_name
        self.var = var
        self.crs = crs
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.ref_vars: List[xr.DataArray] = []

def get_dataset_grid_mapping_templates(dataset: xr.Dataset) \
        -> List[GridMappingTemplate]:
    """
    Find grid mappings encoded as described in the CF conventions
    [Horizontal Coordinate Reference Systems, Grid Mappings, and Projections]
    (http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#grid-mappings-and-projections).

    :param dataset: The dataset
    :return:
    """

    grid_mappings = _get_raw_grid_mappings(dataset)

    for gm in grid_mappings.values():
        if gm.ref_vars:
            ref_var = gm.ref_vars[0]
            x_dim, y_dim = ref_var.dims[-1], ref_var.dims[-2],
            x_coors, y_coords, gms = _get_xy_coords(dataset, x_dim, y_dim)
            gm.x_coords, gm.y_coords = x_coors, y_coords


def _get_xy_coords(dataset: xr.Dataset,
                   x_dim: Hashable,
                   y_dim: Hashable) \
        -> Optional[Tuple[xr.DataArray,
                          xr.DataArray,
                          Optional[GridMappingSchema]]]:
    x_dims = (x_dim,)
    y_dims = (y_dim,)
    yx_dims = (y_dim, x_dim)

    # 1D-coordinate variables named after their dimensions
    if x_dim in dataset.coords and y_dim in dataset.coords:
        x_coords = dataset.coords[x_dim]
        y_coords = dataset.coords[y_dim]
        if x_coords.dims == x_dims and y_coords.dims == y_dims:
            return x_coords, y_coords, None

    # 1D-coordinate variables identified by "standard_name"
    for gms in GM_SCHEMAS:
        x_name, y_name = gms.standard_names
        x_coords = None
        y_coords = None
        for var_name, var in dataset.coords.items():
            standard_name = var.attrs.get("standard_name")
            if standard_name == x_name and var.dims == x_dims:
                x_coords = var
            if standard_name == y_name and var.dims == y_dims:
                y_coords = var
        if x_coords is not None and y_coords is not None:
            return x_coords, y_coords, gms

    # 2D-coordinate variables identified by "standard_name"
    for gms in GM_SCHEMAS:
        x_name, y_name = gms.standard_names
        x_coords = None
        y_coords = None
        for var_name, var in dataset.data_vars.items():
            standard_name = var.attrs.get("standard_name")
            if standard_name == x_name and var.dims == yx_dims:
                x_coords = var
            if standard_name == y_name and var.dims == yx_dims:
                y_coords = var
        if x_coords is not None and y_coords is not None:
            return x_coords, y_coords, gms

    # 1D-coordinate variables identified by common variable names
    for gms in GM_SCHEMAS:
        for x_name, y_name in gms.common_var_names:
            x_coords = dataset.get(x_name)
            y_coords = dataset.get(y_name)
            if x_coords is not None and y_coords is not None \
                    and x_coords.dims == x_dims \
                    and y_coords.dims == y_dims:
                return x_coords, y_coords, gms

    # 2D-coordinate variables identified by common variable names
    for gms in GM_SCHEMAS:
        for x_name, y_name in gms.common_var_names:
            x_coords = dataset.get(x_name)
            y_coords = dataset.get(y_name)
            if x_coords is not None and y_coords is not None \
                    and x_coords.dims == yx_dims \
                    and y_coords.dims == yx_dims:
                return x_coords, y_coords, gms

    return None

def _get_raw_grid_mappings(dataset):
    grid_mappings = dict()
    # Find any grid mappings by parsing variable attributes to CRS
    #
    for var_name, var in dataset.data_vars.items():
        try:
            crs = pyproj.crs.CRS.from_cf(var.attrs)
        except pyproj.crs.CRSError:
            continue
        grid_mappings[var_name] = GridMappingTemplate(var, crs)
    # Add spatial variables to corresponding grid mapping
    # by their CF 'grid_mapping' attribute
    #
    for var_name, var in dataset.data_vars.items():
        if var_name in grid_mappings:
            continue
        if var.ndim < 2:
            continue
        gm_var_name = var.attrs.get('grid_mapping')
        if not gm_var_name or gm_var_name not in grid_mappings:
            continue
        gm = grid_mappings[gm_var_name]
        gm.ref_vars.append(var)

    if len(grid_mappings) > 1:
        # We have more than one grid mapping.
        # Just keep the referenced grid mappings.
        ref_grid_mappings = filter(lambda gm: bool(gm.ref_var_names),
                                   grid_mappings.values())
        if ref_grid_mappings:
            grid_mappings = {gm.var_name: gm for gm in ref_grid_mappings}
        else:
            # There are no referenced grid mappings.
            # So we cannot unambiguously tell which variable uses which
            # grid mappings. Therefore, we choose any:
            gm = next(iter(grid_mappings.values()))
            grid_mappings = {gm.var_name: gm}

    return grid_mappings


def _complement_grid_mapping_coords(
        coords: GridCoords,
        grid_mapping_name: Optional[str],
        missing_crs: Optional[pyproj.crs.CRS],
        grid_mappings: Dict[Optional[str], GridMappingProxy]
):
    if coords.x is not None or coords.y is not None:
        grid_mapping = next((grid_mapping
                             for grid_mapping in grid_mappings.values()
                             if grid_mapping_name is None
                             or grid_mapping_name == grid_mapping.name),
                            None)
        if grid_mapping is None and missing_crs is not None:
            grid_mapping = GridMappingProxy(crs=missing_crs,
                                            name=grid_mapping_name)
            grid_mappings[None] = grid_mapping

        if grid_mapping is not None:
            if grid_mapping.coords is None:
                grid_mapping.coords = coords
            # Edge case from GeoTIFF with CRS-84 with 1D
            # coordinates named "x" and "y" as read by rioxarray
            if grid_mapping.coords.x is None:
                grid_mapping.coords.x = coords.x
            if grid_mapping.coords.y is None:
                grid_mapping.coords.y = coords.y


def _find_potential_coord_vars(dataset: xr.Dataset) -> List[Hashable]:
    """
    Find potential coordinate variables.

    We need this function as we can not rely on xarray.Dataset.coords,
    because 2D coordinate arrays are most likely not indicated as such
    in many datasets.
    """

    # Collect bounds variables. We must exclude them.
    bounds_vars = set()
    for k in dataset.variables:
        var = dataset[k]

        # Bounds variable as recommended through CF conventions
        bounds_k = var.attrs.get('bounds')
        if bounds_k is not None and bounds_k in dataset:
            bounds_vars.add(bounds_k)

        # Bounds variable by naming convention,
        # e.g. "lon_bnds" or "lat_bounds"
        k_splits = str(k).rsplit("_", maxsplit=1)
        if len(k_splits) == 2:
            k_base, k_suffix = k_splits
            if k_suffix in ('bnds', 'bounds') and k_base in dataset:
                bounds_vars.add(k)

    potential_coord_vars = []

    # First consider any CF global attribute "coordinates"
    coordinates = dataset.attrs.get('coordinates')
    if coordinates is not None:
        for var_name in coordinates.split():
            if _is_potential_coord_var(dataset, bounds_vars, var_name):
                potential_coord_vars.append(var_name)

    # Then consider any other 1D/2D variables
    for var_name in dataset.variables:
        if var_name not in potential_coord_vars \
                and _is_potential_coord_var(dataset, bounds_vars, var_name):
            potential_coord_vars.append(var_name)

    return potential_coord_vars


def _is_potential_coord_var(dataset: xr.Dataset,
                            bounds_var_names: Set[str],
                            var_name: Hashable) -> bool:
    if var_name in dataset:
        var = dataset[var_name]
        return var.ndim in (1, 2) and var_name not in bounds_var_names
    return False


def _find_dataset_tile_size(dataset: xr.Dataset,
                            x_dim_name: Hashable,
                            y_dim_name: Hashable) \
        -> Optional[Tuple[int, int]]:
    """Find the most likely tile size in *dataset*"""
    dataset_chunks = get_dataset_chunks(dataset)
    tile_width = dataset_chunks.get(x_dim_name)
    tile_height = dataset_chunks.get(y_dim_name)
    if tile_width is not None and tile_height is not None:
        return tile_width, tile_height
    return None


def add_spatial_ref(dataset_store: zarr.convenience.StoreLike,
                    crs: pyproj.CRS,
                    crs_var_name: Optional[str] = 'spatial_ref',
                    xy_dim_names: Optional[Tuple[str, str]] = None):
    """
    Helper function that allows adding a spatial reference to an
    existing Zarr dataset.

    :param dataset_store: The dataset's existing Zarr store or path.
    :param crs: The spatial coordinate reference system.
    :param crs_var_name: The name of the variable that will hold the
        spatial reference. Defaults to "spatial_ref".
    :param xy_dim_names: The names of the x and y dimensions.
        Defaults to ("x", "y").
    """
    assert_instance(dataset_store, (MutableMapping, str), name='group_store')
    assert_instance(crs_var_name, str, name='crs_var_name')
    x_dim_name, y_dim_name = xy_dim_names or ('x', 'y')

    spatial_attrs = crs.to_cf()
    spatial_attrs['_ARRAY_DIMENSIONS'] = []  # Required by xarray
    group = zarr.open(dataset_store, mode='r+')
    spatial_ref = group.array(crs_var_name,
                              0,
                              shape=(),
                              dtype=np.uint8,
                              fill_value=0)
    spatial_ref.attrs.update(**spatial_attrs)

    for item_name, item in group.items():
        if item_name != crs_var_name:
            dims = item.attrs.get('_ARRAY_DIMENSIONS')
            if dims and len(dims) >= 2 \
                    and dims[-2] == y_dim_name \
                    and dims[-1] == x_dim_name:
                item.attrs['grid_mapping'] = crs_var_name

    zarr.convenience.consolidate_metadata(dataset_store)
