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

import math
from typing import Dict, List

import numpy as np
import shapely.geometry
import xarray as xr

from ..context import ServiceContext
from ..errors import ServiceBadRequestError
from ..utils import get_dataset_bounds, get_dataset_geometry, get_box_split_bounds_geometry, get_geometry_mask, \
    GeoJSON, timestamp_to_iso_string


def get_time_series_info(ctx: ServiceContext) -> Dict:
    time_series_info = {'layers': []}
    descriptors = ctx.get_dataset_descriptors()
    for descriptor in descriptors:
        if 'Identifier' in descriptor:
            dataset = ctx.get_dataset(descriptor['Identifier'])
            if 'time' not in dataset.variables:
                continue
            xmin, ymin, xmax, ymax = get_dataset_bounds(dataset)
            time_data = dataset.variables['time'].data
            time_stamps = []
            for time in time_data:
                time_stamps.append(timestamp_to_iso_string(time))
            for variable in dataset.data_vars.variables:
                variable_dict = {'name': '{0}.{1}'.format(descriptor['Identifier'], variable),
                                 'dates': time_stamps,
                                 'bounds': dict(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)}
                time_series_info['layers'].append(variable_dict)
    return time_series_info


def get_time_series_for_point(ctx: ServiceContext,
                              ds_name: str, var_name: str,
                              lon: float, lat: float,
                              start_date: np.datetime64 = None,
                              end_date: np.datetime64 = None) -> Dict:
    dataset, variable = ctx.get_dataset_and_variable(ds_name, var_name)
    return _get_time_series_for_point(dataset, variable,
                                      shapely.geometry.Point(lon, lat),
                                      start_date=start_date, end_date=end_date)


def get_time_series_for_geometry(ctx: ServiceContext,
                                 ds_name: str, var_name: str,
                                 geometry: Dict,
                                 start_date: np.datetime64 = None,
                                 end_date: np.datetime64 = None) -> Dict:
    dataset, variable = ctx.get_dataset_and_variable(ds_name, var_name)
    if not GeoJSON.is_geometry(geometry):
        raise ServiceBadRequestError("Invalid GeoJSON geometry")
    if isinstance(geometry, dict):
        geometry = shapely.geometry.shape(geometry)
    return _get_time_series_for_geometry(dataset, variable,
                                         geometry,
                                         start_date=start_date, end_date=end_date)


def get_time_series_for_geometry_collection(ctx: ServiceContext,
                                            ds_name: str, var_name: str,
                                            geometry_collection: Dict,
                                            start_date: np.datetime64 = None,
                                            end_date: np.datetime64 = None) -> Dict:
    dataset, variable = ctx.get_dataset_and_variable(ds_name, var_name)
    geometries = GeoJSON.get_geometry_collection_geometries(geometry_collection)
    if geometries is None:
        raise ServiceBadRequestError("Invalid GeoJSON geometry collection")
    shapes = []
    for geometry in geometries:
        try:
            geometry = shapely.geometry.shape(geometry)
        except (TypeError, ValueError) as e:
            raise ServiceBadRequestError("Invalid GeoJSON geometry collection") from e
        shapes.append(geometry)
    return _get_time_series_for_geometries(dataset, variable, shapes, start_date, end_date)


def get_time_series_for_feature_collection(ctx: ServiceContext,
                                           ds_name: str, var_name: str,
                                           feature_collection: Dict,
                                           start_date: np.datetime64 = None,
                                           end_date: np.datetime64 = None) -> Dict:
    dataset, variable = ctx.get_dataset_and_variable(ds_name, var_name)
    features = GeoJSON.get_feature_collection_features(feature_collection)
    if features is None:
        raise ServiceBadRequestError("Invalid GeoJSON feature collection")
    shapes = []
    for feature in features:
        geometry = GeoJSON.get_feature_geometry(feature)
        try:
            geometry = shapely.geometry.shape(geometry)
        except (TypeError, ValueError) as e:
            raise ServiceBadRequestError("Invalid GeoJSON feature collection") from e
        shapes.append(geometry)
    return _get_time_series_for_geometries(dataset, variable, shapes, start_date, end_date)


def _get_time_series_for_point(dataset: xr.Dataset,
                               variable: xr.DataArray,
                               point: shapely.geometry.Point,
                               start_date: np.datetime64 = None,
                               end_date: np.datetime64 = None) -> Dict:
    bounds = get_dataset_geometry(dataset)
    if not bounds.contains(point):
        return {'results': []}

    point_subset = variable.sel(lon=point.x, lat=point.y, method='Nearest')
    # noinspection PyTypeChecker
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
    return {'results': time_series}


def _get_time_series_for_geometry(dataset: xr.Dataset,
                                  variable: xr.DataArray,
                                  geometry: shapely.geometry.base.BaseGeometry,
                                  start_date: np.datetime64 = None,
                                  end_date: np.datetime64 = None) -> Dict:
    if isinstance(geometry, shapely.geometry.Point):
        return _get_time_series_for_point(dataset, variable,
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
    subset_variable = ds_subset[variable.name]
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
        mean_ts_var = variable_slice.mean(["lat", "lon"]).values.item()

        statistics = {'totalCount': total_count}
        if np.isnan(mean_ts_var):
            statistics['validCount'] = 0
            statistics['average'] = None
        else:
            statistics['validCount'] = valid_count
            statistics['average'] = float(mean_ts_var)
        result = {'result': statistics, 'date': timestamp_to_iso_string(variable_slice.time.data)}
        time_series.append(result)

    return {'results': time_series}


def _get_time_series_for_geometries(dataset: xr.Dataset,
                                    variable: xr.DataArray,
                                    geometries: List[shapely.geometry.base.BaseGeometry],
                                    start_date: np.datetime64 = None,
                                    end_date: np.datetime64 = None) -> Dict:
    time_series = []
    for geometry in geometries:
        result = _get_time_series_for_geometry(dataset, variable,
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
