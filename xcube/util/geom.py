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

from typing import Optional, Union, Dict, Tuple

import affine
import math
import numpy as np
import rasterio.features
import shapely.geometry
import shapely.geometry
import xarray as xr

Bounds = Tuple[float, float, float, float]
SplitBounds = Tuple[Bounds, Optional[Bounds]]


def where_geometry(dataset: xr.Dataset,
                   geometry: shapely.geometry.base.BaseGeometry,
                   mask_var_name: str = None) -> Optional[xr.Dataset]:
    ds_lon_min, ds_lat_min, ds_lon_max, ds_lat_max = get_dataset_bounds(dataset)
    inv_y = float(dataset.lat[0]) < float(dataset.lat[-1])
    dataset_geometry = get_box_split_bounds_geometry(ds_lon_min, ds_lat_min, ds_lon_max, ds_lat_max)
    # TODO: split geometry
    split_geometry = geometry
    actual_geometry = dataset_geometry.intersection(split_geometry)
    if actual_geometry.is_empty:
        return None

    width = len(dataset.lon)
    height = len(dataset.lat)
    res = (ds_lat_max - ds_lat_min) / height

    g_lon_min, g_lat_min, g_lon_max, g_lat_max = actual_geometry.bounds
    x1 = _clamp(int(math.floor((g_lon_min - ds_lon_min) / res)), 0, width - 1)
    x2 = _clamp(int(math.ceil((g_lon_max - ds_lon_min) / res)) + 1, 0, width - 1)
    y1 = _clamp(int(math.floor((g_lat_min - ds_lat_min) / res)), 0, height - 1)
    y2 = _clamp(int(math.ceil((g_lat_max - ds_lat_min) / res)) + 1, 0, height - 1)
    if not inv_y:
        _y1, _y2 = y1, y2
        y1 = height - _y2 - 1
        y2 = height - _y1 - 1
    ds_subset = dataset.isel(lon=slice(x1, x2), lat=slice(y1, y2))
    subset_ds_lon_min, subset_ds_lat_min, subset_ds_lon_max, subset_ds_lat_max = get_dataset_bounds(ds_subset)
    subset_width = len(ds_subset.lon)
    subset_height = len(ds_subset.lat)

    mask_data = get_geometry_mask(subset_width, subset_height, actual_geometry, subset_ds_lon_min, subset_ds_lat_min,
                                  res)
    mask = xr.DataArray(mask_data, coords=dict(lat=ds_subset.lat, lon=ds_subset.lon), dims=('lat', 'lon'))

    ds_subset_masked = xr.where(mask, ds_subset, np.nan)
    if mask_var_name:
        ds_subset_masked[mask_var_name] = mask

    return ds_subset_masked


def get_geometry_mask(width: int, height: int,
                      geometry: Union[shapely.geometry.base.BaseGeometry, Dict],
                      lon_min: float, lat_min: float, res: float) -> np.ndarray:
    # noinspection PyTypeChecker
    transform = affine.Affine(res, 0.0, lon_min,
                              0.0, -res, lat_min + res * height)
    return rasterio.features.geometry_mask([geometry],
                                           out_shape=(height, width),
                                           transform=transform,
                                           all_touched=True,
                                           invert=True)


def get_dataset_geometry(dataset: Union[xr.Dataset, xr.DataArray]) -> shapely.geometry.base.BaseGeometry:
    return get_box_split_bounds_geometry(*get_dataset_bounds(dataset))


def get_dataset_bounds(dataset: Union[xr.Dataset, xr.DataArray]) -> Bounds:
    lon_var = dataset.coords.get("lon")
    lat_var = dataset.coords.get("lat")
    if lon_var is None:
        raise ValueError('Missing coordinate variable "lon"')
    if lat_var is None:
        raise ValueError('Missing coordinate variable "lat"')

    lon_bnds_name = lon_var.attrs["bounds"] if "bounds" in lon_var.attrs else "lon_bnds"
    if lon_bnds_name in dataset.coords:
        lon_bnds_var = dataset.coords[lon_bnds_name]
        lon_min = lon_bnds_var[0][0]
        lon_max = lon_bnds_var[-1][1]
    else:
        lon_min = lon_var[0]
        lon_max = lon_var[-1]
        delta = min(abs(np.diff(lon_var)))
        lon_min -= 0.5 * delta
        lon_max += 0.5 * delta

    lat_bnds_name = lat_var.attrs["bounds"] if "bounds" in lat_var.attrs else "lat_bnds"
    if lat_bnds_name in dataset.coords:
        lat_bnds_var = dataset.coords[lat_bnds_name]
        lat1 = lat_bnds_var[0][0]
        lat2 = lat_bnds_var[-1][1]
        lat_min = min(lat1, lat2)
        lat_max = max(lat1, lat2)
    else:
        lat1 = lat_var[0]
        lat2 = lat_var[-1]
        delta = min(abs(np.diff(lat_var)))
        lat_min = min(lat1, lat2) - 0.5 * delta
        lat_max = max(lat1, lat2) + 0.5 * delta

    return float(lon_min), float(lat_min), float(lon_max), float(lat_max)


def get_box_split_bounds(lon_min: float, lat_min: float,
                         lon_max: float, lat_max: float) -> SplitBounds:
    if lon_max >= lon_min:
        return (lon_min, lat_min, lon_max, lat_max), None
    else:
        return (lon_min, lat_min, 180.0, lat_max), (-180.0, lat_min, lon_max, lat_max)


def get_box_split_bounds_geometry(lon_min: float, lat_min: float,
                                  lon_max: float, lat_max: float) -> shapely.geometry.base.BaseGeometry:
    box_1, box_2 = get_box_split_bounds(lon_min, lat_min, lon_max, lat_max)
    if box_2 is not None:
        return shapely.geometry.MultiPolygon(polygons=[shapely.geometry.box(*box_1), shapely.geometry.box(*box_2)])
    else:
        return shapely.geometry.box(*box_1)


def _clamp(x, x1, x2):
    if x < x1:
        return x1
    if x > x2:
        return x2
    return x
