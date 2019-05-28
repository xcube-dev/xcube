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

from typing import Optional, Sequence, Union, Dict, Any

import math
import numpy as np
import shapely.geometry
import xarray as xr

from ..api.select import select_vars
from ..webapi.utils import get_dataset_bounds, get_box_split_bounds_geometry, get_geometry_mask, \
    GeoJSON


def where_geometry(dataset: xr.Dataset,
                   geometry: Union[shapely.geometry.base.BaseGeometry, Dict[str, Any]],
                   var_names: Sequence[str] = None,
                   mask_var_name: str = None) -> Optional[xr.Dataset]:
    if isinstance(geometry, dict):
        if not GeoJSON.is_geometry(geometry):
            raise ValueError("Invalid GeoJSON geometry")
        geometry = shapely.geometry.shape(geometry)

    ds_lon_min, ds_lat_min, ds_lon_max, ds_lat_max = get_dataset_bounds(dataset)
    inv_y = float(dataset.lat[0]) < float(dataset.lat[-1])
    dataset_geometry = get_box_split_bounds_geometry(ds_lon_min, ds_lat_min, ds_lon_max, ds_lat_max)
    # TODO: split geometry
    split_geometry = geometry
    actual_geometry = dataset_geometry.intersection(split_geometry)
    if actual_geometry.is_empty:
        return None

    dataset = select_vars(dataset, var_names)
    if len(dataset.data_vars) == 0:
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


def _clamp(x, x1, x2):
    if x < x1:
        return x1
    if x > x2:
        return x2
    return x
