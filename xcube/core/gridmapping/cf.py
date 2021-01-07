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
from typing import Optional, Dict, Any, Hashable, Union

import pyproj
import xarray as xr


class GridCoords:
    def __init__(self,
                 x: xr.DataArray = None,
                 y: xr.DataArray = None):
        self.x = x
        self.y = y


class GridMapping:
    def __init__(self,
                 crs: pyproj.crs.CRS,
                 name: str = None,
                 coords: GridCoords = None):
        self.crs = crs
        self.name = name
        self.coords = coords


def get_dataset_grid_mappings(dataset: xr.Dataset, *,
                              missing_latitude_longitude_crs: pyproj.crs.CRS = None,
                              missing_rotated_latitude_longitude_crs: pyproj.crs.CRS = None,
                              missing_projected_crs: pyproj.crs.CRS = None,
                              emit_warnings: bool = False) -> Dict[Union[Hashable, None], GridMapping]:
    # Find any grid mapping variables
    #
    grid_mappings = dict()
    for k, var in dataset.variables.items():
        grid_mapping = _parse_crs_from_attrs(var.attrs)
        if grid_mapping is not None:
            grid_mappings[k] = grid_mapping

    # If no grid mapping variables found,
    # try if CRS is encoded in dataset attributes
    #
    if not grid_mappings:
        grid_mapping = _parse_crs_from_attrs(dataset.attrs)
        if grid_mapping is not None:
            grid_mappings[None] = grid_mapping

    # Find coordinate variables that use a CF standard_name.
    #
    latitude_longitude_coords = GridCoords(x=None, y=None)
    rotated_latitude_longitude_coords = GridCoords(x=None, y=None)
    projected_coords = GridCoords(x=None, y=None)
    for k, var in dataset.coords.items():
        standard_name = var.attrs.get('standard_name')
        if standard_name == 'longitude':
            latitude_longitude_coords.x = var
        elif standard_name == 'latitude':
            latitude_longitude_coords.y = var
        elif standard_name == 'grid_longitude':
            rotated_latitude_longitude_coords.x = var
        elif standard_name == 'grid_latitude':
            rotated_latitude_longitude_coords.y = var
        elif standard_name == 'projection_x_coordinate':
            projected_coords.x = var
        elif standard_name == 'projection_y_coordinate':
            projected_coords.y = var

    # Find coordinate variables by common naming convention.
    #
    for k, var in dataset.coords.items():
        if latitude_longitude_coords.x is None \
                and k in {'lon', 'longitude'}:
            latitude_longitude_coords.x = var
        elif latitude_longitude_coords.y is None \
                and k in {'lat', 'latitude'}:
            latitude_longitude_coords.y = var
        elif rotated_latitude_longitude_coords.x is None \
                and k in {'rlon', 'rlongitude'}:
            rotated_latitude_longitude_coords.x = var
        elif rotated_latitude_longitude_coords.y is None \
                and k in {'rlat', 'rlatitude'}:
            rotated_latitude_longitude_coords.y = var
        elif projected_coords.x is None \
                and k in {'x', 'xc'}:
            projected_coords.x = var
        elif projected_coords.y is None \
                and k in {'y', 'yc'}:
            projected_coords.y = var

    # Assign found coordinates to grid mappings
    #
    for k, grid_mapping in grid_mappings.items():
        if grid_mapping.name == 'latitude_longitude':
            grid_mapping.coords = latitude_longitude_coords
        elif grid_mapping.name == 'rotated_latitude_longitude':
            grid_mapping.coords = rotated_latitude_longitude_coords
        else:
            grid_mapping.coords = projected_coords

    _complement_grid_mapping_coords(latitude_longitude_coords,
                                    'latitude_longitude',
                                    missing_latitude_longitude_crs
                                    or pyproj.crs.CRS(4326),
                                    grid_mappings)
    _complement_grid_mapping_coords(rotated_latitude_longitude_coords,
                                    'rotated_latitude_longitude',
                                    missing_rotated_latitude_longitude_crs,
                                    grid_mappings)
    _complement_grid_mapping_coords(projected_coords,
                                    None,
                                    missing_projected_crs,
                                    grid_mappings)

    # Collect complete grid mappings
    complete_grid_mappings = dict()
    for k, grid_mapping in grid_mappings.items():
        if grid_mapping.coords is not None \
                and grid_mapping.coords.x is not None \
                and grid_mapping.coords.y is not None:
            # TODO: add more consistency checks, eg both 1d or both 2d with same dims
            complete_grid_mappings[k] = grid_mapping
        elif emit_warnings:
            warnings.warn(f'CRS "{grid_mapping.name}": '
                          f'missing x- and/or y-coordinates '
                          f'(grid mapping variable "{k}": '
                          f'grid_mapping_name="{grid_mapping.name}")')

    return complete_grid_mappings


def _parse_crs_from_attrs(attrs: Dict[Hashable, Any]) -> Optional[GridMapping]:
    # noinspection PyBroadException
    try:
        crs = pyproj.crs.CRS.from_cf(attrs)
    except pyproj.crs.CRSError:
        return None
    return GridMapping(crs=crs, name=attrs.get('grid_mapping_name'), coords=None)


def _complement_grid_mapping_coords(coords: GridCoords,
                                    grid_mapping_name: Optional[str],
                                    missing_crs: Optional[pyproj.crs.CRS],
                                    grid_mappings: Dict[Optional[str], GridMapping]):
    if coords.x is not None or coords.y is not None:
        grid_mapping = next((grid_mapping
                             for grid_mapping in grid_mappings.values()
                             if grid_mapping.name == grid_mapping_name), None)
        if grid_mapping is None:
            grid_mapping = grid_mappings.get(None)
            if grid_mapping is None and missing_crs is not None:
                grid_mapping = GridMapping(crs=missing_crs,
                                           name=grid_mapping_name,
                                           coords=None)
                grid_mappings[None] = grid_mapping

        if grid_mapping is not None and grid_mapping.coords is None:
            grid_mapping.coords = coords
