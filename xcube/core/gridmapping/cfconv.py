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


class GridCoords:
    """
    Grid coordinates comprising x and y of
    type xarray.DataArray.
    """

    def __init__(self):
        self.x: Optional[xr.DataArray] = None
        self.y: Optional[xr.DataArray] = None


class GridMappingProxy:
    """
    Grid mapping comprising *crs* of type pyproj.crs.CRS,
    grid coordinates, an optional name, coordinates, and a
    tile size (= spatial chunk sizes).
    """

    def __init__(self,
                 crs: pyproj.crs.CRS,
                 name: Optional[str] = None):
        self.crs: pyproj.crs.CRS = crs
        self.name: Optional[str] = name
        self.coords: Optional[xr.DataArray] = None
        self.tile_size: Optional[Tuple[int, int]] = None


def get_dataset_grid_mapping_proxies(
        dataset: xr.Dataset,
        *,
        missing_latitude_longitude_crs: pyproj.crs.CRS = None,
        missing_rotated_latitude_longitude_crs: pyproj.crs.CRS = None,
        missing_projected_crs: pyproj.crs.CRS = None,
        emit_warnings: bool = False
) -> Dict[Union[Hashable, None], GridMappingProxy]:
    """
    Find grid mappings encoded as described in the CF conventions
    [Horizontal Coordinate Reference Systems, Grid Mappings, and Projections]
    (http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#grid-mappings-and-projections).

    :param dataset:
    :param missing_latitude_longitude_crs:
    :param missing_rotated_latitude_longitude_crs:
    :param missing_projected_crs:
    :param emit_warnings:
    :return:
    """
    # Find any grid mapping variables
    #
    grid_mapping_proxies = dict()
    for var_name, var in dataset.variables.items():
        gmp = _parse_crs_from_attrs(var.attrs)
        if gmp is not None:
            grid_mapping_proxies[var_name] = gmp

    # If no grid mapping variables found,
    # try if CRS is encoded in dataset attributes
    #
    if not grid_mapping_proxies:
        gmp = _parse_crs_from_attrs(dataset.attrs)
        if gmp is not None:
            grid_mapping_proxies[None] = gmp

    # Find coordinate variables.
    #

    latitude_longitude_coords = GridCoords()
    rotated_latitude_longitude_coords = GridCoords()
    projected_coords = GridCoords()

    potential_coord_vars = _find_potential_coord_vars(dataset)

    # Find coordinate variables that use a CF standard_name.
    #
    coords_standard_names = (
        (latitude_longitude_coords,
         'longitude', 'latitude'),
        (rotated_latitude_longitude_coords,
         'grid_longitude', 'grid_latitude'),
        (projected_coords,
         'projection_x_coordinate', 'projection_y_coordinate')
    )
    for var_name in potential_coord_vars:
        var = dataset[var_name]
        if var.ndim not in (1, 2):
            continue
        standard_name = var.attrs.get('standard_name')
        for coords, x_name, y_name in coords_standard_names:
            if standard_name == x_name:
                coords.x = var
            if standard_name == y_name:
                coords.y = var

    # Find coordinate variables by common naming convention.
    #
    coords_var_names = (
        (latitude_longitude_coords,
         {'lon', 'longitude'}, {'lat', 'latitude'}),
        (rotated_latitude_longitude_coords,
         {'rlon', 'rlongitude'}, {'rlat', 'rlatitude'}),
        (projected_coords,
         {'x', 'xc', 'transformed_x'}, {'y', 'yc', 'transformed_y'})
    )
    for var_name in potential_coord_vars:
        var = dataset[var_name]
        if var.ndim not in (1, 2):
            continue
        for coords, x_names, y_names in coords_var_names:
            if var_name in x_names:
                coords.x = var
            if var_name in y_names:
                coords.y = var

    # Assign found coordinates to grid mappings
    #
    for gmp in grid_mapping_proxies.values():
        if gmp.name == 'latitude_longitude':
            gmp.coords = latitude_longitude_coords
        elif gmp.name == 'rotated_latitude_longitude':
            gmp.coords = rotated_latitude_longitude_coords
        else:
            gmp.coords = projected_coords

    _complement_grid_mapping_coords(latitude_longitude_coords,
                                    'latitude_longitude',
                                    missing_latitude_longitude_crs
                                    or pyproj.crs.CRS(4326),
                                    grid_mapping_proxies)
    _complement_grid_mapping_coords(rotated_latitude_longitude_coords,
                                    'rotated_latitude_longitude',
                                    missing_rotated_latitude_longitude_crs,
                                    grid_mapping_proxies)
    _complement_grid_mapping_coords(projected_coords,
                                    None,
                                    missing_projected_crs,
                                    grid_mapping_proxies)

    # Collect complete grid mappings
    complete_grid_mappings = dict()
    for var_name, gmp in grid_mapping_proxies.items():
        if gmp.coords is not None \
                and gmp.coords.x is not None \
                and gmp.coords.y is not None \
                and gmp.coords.x.size >= 2 \
                and gmp.coords.y.size >= 2 \
                and gmp.coords.x.ndim == gmp.coords.y.ndim:
            if gmp.coords.x.ndim == 1:
                gmp.tile_size = _find_dataset_tile_size(
                    dataset,
                    gmp.coords.x.dims[0],
                    gmp.coords.y.dims[0]
                )
                complete_grid_mappings[var_name] = gmp
            elif gmp.coords.x.ndim == 2 \
                    and gmp.coords.x.dims == gmp.coords.y.dims:
                gmp.tile_size = _find_dataset_tile_size(
                    dataset,
                    gmp.coords.x.dims[1],
                    gmp.coords.x.dims[0]
                )
                complete_grid_mappings[var_name] = gmp
        elif emit_warnings:
            warnings.warn(f'CRS "{gmp.name}": '
                          f'missing x- and/or y-coordinates '
                          f'(grid mapping variable "{var_name}": '
                          f'grid_mapping_name="{gmp.name}")')

    return complete_grid_mappings


def _parse_crs_from_attrs(attrs: Dict[Hashable, Any]) \
        -> Optional[GridMappingProxy]:
    # noinspection PyBroadException
    try:
        crs = pyproj.crs.CRS.from_cf(attrs)
    except pyproj.crs.CRSError:
        return None
    return GridMappingProxy(crs=crs,
                            name=attrs.get('grid_mapping_name'))


def _complement_grid_mapping_coords(
        coords: GridCoords,
        grid_mapping_name: Optional[str],
        missing_crs: Optional[pyproj.crs.CRS],
        grid_mappings: Dict[Optional[str], GridMappingProxy]
):
    if coords.x is not None or coords.y is not None:
        grid_mapping = next((grid_mapping
                             for grid_mapping in grid_mappings.values()
                             if grid_mapping.name == grid_mapping_name), None)
        if grid_mapping is None:
            grid_mapping = grid_mappings.get(None)
            if grid_mapping is None and missing_crs is not None:
                grid_mapping = GridMappingProxy(crs=missing_crs,
                                                name=grid_mapping_name)
                grid_mappings[None] = grid_mapping

        if grid_mapping is not None and grid_mapping.coords is None:
            grid_mapping.coords = coords


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
        bounds_k = var.attrs.get('bounds')
        if bounds_k is not None and bounds_k in dataset:
            bounds_vars.add(bounds_k)

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
