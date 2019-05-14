# The MIT License (MIT)
# Copyright (c) 2018 by the xcube development team and contributors
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

from typing import Dict, List, Tuple, Union

import math
import numpy as np
import shapely.geometry
import xarray as xr

from ..webapi.utils import get_dataset_bounds, get_dataset_geometry, get_box_split_bounds_geometry, get_geometry_mask, \
    timestamp_to_iso_string


def get_time_series_for_point(dataset: xr.Dataset,
                              var_name: str,
                              point: shapely.geometry.Point,
                              start_date: np.datetime64 = None,
                              end_date: np.datetime64 = None) -> Dict:
    bounds = get_dataset_geometry(dataset)
    if not bounds.contains(point):
        return {'results': []}

    variable = dataset.data_vars[var_name]
    point_subset = variable.sel(lon=point.x, lat=point.y, method='Nearest')
    time_subset = point_subset.sel(time=slice(start_date, end_date))
    time_series = []
    for entry in time_subset:
        statistics = {'totalCount': 1}
        item = entry.values.item()
        if np.isnan(item):
            statistics['validCount'] = 0
            statistics['average'] = None
        else:
            statistics['validCount'] = 1
            statistics['average'] = item
        result = {'result': statistics, 'date': timestamp_to_iso_string(entry.time.data)}
        time_series.append(result)

    ancillary_var_name, ancillary_field_name = _find_ancillary_var_name(dataset, variable)
    if ancillary_var_name is not None:
        ancillary_variable = dataset.data_vars[ancillary_var_name]
        ancillary_point_subset = ancillary_variable.sel(lon=point.x, lat=point.y, method='Nearest')
        ancillary_time_subset = ancillary_point_subset.sel(time=slice(start_date, end_date))
        num_time_steps = len(time_series)
        if len(ancillary_time_subset) == num_time_steps:
            for index, entry in zip(range(num_time_steps), ancillary_time_subset):
                ancillary_value = entry.values.item()
                statistics = time_series[index]['result']
                statistics[ancillary_field_name] = None if np.isnan(ancillary_value) else ancillary_value

    return {'results': time_series}


def _get_time_series_for_geometry(dataset: xr.Dataset,
                                  var_name: str,
                                  geometry: shapely.geometry.base.BaseGeometry,
                                  start_date: np.datetime64 = None,
                                  end_date: np.datetime64 = None) -> Dict:
    if isinstance(geometry, shapely.geometry.Point):
        return get_time_series_for_point(dataset, var_name,
                                         geometry,
                                         start_date=start_date, end_date=end_date)

    ds_lon_min, ds_lat_min, ds_lon_max, ds_lat_max = get_dataset_bounds(dataset)
    dataset_geometry = get_box_split_bounds_geometry(ds_lon_min, ds_lat_min, ds_lon_max, ds_lat_max)
    # TODO: split geometry
    split_geometry = geometry
    actual_geometry = dataset_geometry.intersection(split_geometry)
    if actual_geometry.is_empty:
        return {'results': []}

    width = len(dataset.lon)
    height = len(dataset.lat)
    res = (ds_lat_max - ds_lat_min) / height

    g_lon_min, g_lat_min, g_lon_max, g_lat_max = actual_geometry.bounds
    x1 = _clamp(int(math.floor((g_lon_min - ds_lon_min) / res)), 0, width - 1)
    x2 = _clamp(int(math.ceil((g_lon_max - ds_lon_min) / res)) + 1, 0, width - 1)
    y1 = _clamp(int(math.floor((ds_lat_max - g_lat_max) / res)), 0, height - 1)
    y2 = _clamp(int(math.ceil((ds_lat_max - g_lat_min) / res)) + 1, 0, height - 1)
    ds_subset = dataset.isel(lon=slice(x1, x2), lat=slice(y1, y2))
    ds_subset = ds_subset.sel(time=slice(start_date, end_date))
    subset_ds_lon_min, subset_ds_lat_min, subset_ds_lon_max, subset_ds_lat_max = get_dataset_bounds(ds_subset)
    subset_variable = ds_subset[var_name]
    subset_width = len(ds_subset.lon)
    subset_height = len(ds_subset.lat)

    mask = get_geometry_mask(subset_width, subset_height, actual_geometry, subset_ds_lon_min, subset_ds_lat_min, res)
    total_count = np.count_nonzero(mask)
    variable = subset_variable.sel(time=slice(start_date, end_date))
    num_times = len(variable.time)

    time_series = []
    for time_index in range(num_times):
        variable_slice = variable.isel(time=time_index)
        masked_var = variable_slice.where(mask)
        valid_count = len(np.where(np.isfinite(masked_var))[0])
        mean_ts_var = masked_var.mean(["lat", "lon"]).values.item()

        statistics = {'totalCount': total_count}
        if np.isnan(mean_ts_var):
            statistics['validCount'] = 0
            statistics['average'] = None
        else:
            statistics['validCount'] = valid_count
            statistics['average'] = float(mean_ts_var)
        result = {'result': statistics, 'date': timestamp_to_iso_string(variable_slice.time.data)}
        time_series.append(result)

    ancillary_var_name, ancillary_field_name = _find_ancillary_var_name(ds_subset, subset_variable)
    if ancillary_var_name is not None:
        ancillary_variable = ds_subset.data_vars[ancillary_var_name]
        for time_index in range(num_times):
            ancillary_slice = ancillary_variable.isel(time=time_index)
            ancillary_masked = ancillary_slice.where(mask)
            ancillary_mean = ancillary_masked.mean(["lat", "lon"])
            ancillary_value = ancillary_mean.values.item()
            statistics = time_series[time_index]["result"]
            statistics[ancillary_field_name] = None if np.isnan(ancillary_value) else ancillary_value

    return {'results': time_series}


def _get_time_series_for_geometries(dataset: xr.Dataset,
                                    var_name: str,
                                    geometries: List[shapely.geometry.base.BaseGeometry],
                                    start_date: np.datetime64 = None,
                                    end_date: np.datetime64 = None) -> Dict:
    time_series = []
    for geometry in geometries:
        result = _get_time_series_for_geometry(dataset, var_name,
                                               geometry,
                                               start_date=start_date, end_date=end_date)
        time_series.append(result["results"])
    return {'results': time_series}



def _clamp(x, x1, x2):
    if x < x1:
        return x1
    if x > x2:
        return x2
    return x
