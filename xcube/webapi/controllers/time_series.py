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

from typing import Dict, List

import numpy as np
import shapely.geometry
import xarray as xr

from ..context import ServiceContext
from ..errors import ServiceBadRequestError
from ...api import ts
from ...util.ancvar import find_ancillary_var_names
from ...util.geojson import GeoJSON
from ...util.geom import get_dataset_bounds
from ...util.perf import measure_time_cm
from ...util.timecoord import timestamp_to_iso_string


def get_time_series_info(ctx: ServiceContext) -> Dict:
    time_series_info = {'layers': []}
    descriptors = ctx.get_dataset_descriptors()
    for descriptor in descriptors:
        if 'Identifier' in descriptor:
            if descriptor.get('Hidden'):
                continue
            dataset = ctx.get_dataset(descriptor['Identifier'])
            if 'time' not in dataset.variables:
                continue
            xmin, ymin, xmax, ymax = get_dataset_bounds(dataset)
            time_data = dataset.variables['time'].data
            time_stamps = []
            for time in time_data:
                time_stamps.append(timestamp_to_iso_string(time))
            var_names = sorted(dataset.data_vars)
            for var_name in var_names:
                ds_id = descriptor['Identifier']
                variable_dict = {'name': f'{ds_id}.{var_name}',
                                 'dates': time_stamps,
                                 'bounds': dict(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)}
                time_series_info['layers'].append(variable_dict)
    return time_series_info


def get_time_series_for_point(ctx: ServiceContext,
                              ds_name: str, var_name: str,
                              lon: float, lat: float,
                              start_date: np.datetime64 = None,
                              end_date: np.datetime64 = None) -> Dict:
    measure_time = measure_time_cm(disabled=not ctx.trace_perf)
    with measure_time('get_time_series_for_point'):
        dataset = _get_time_series_dataset(ctx, ds_name, var_name)
        return _get_time_series_for_point(dataset, var_name,
                                          shapely.geometry.Point(lon, lat),
                                          start_date=start_date, end_date=end_date)


def get_time_series_for_geometry(ctx: ServiceContext,
                                 ds_name: str, var_name: str,
                                 geometry: Dict,
                                 start_date: np.datetime64 = None,
                                 end_date: np.datetime64 = None) -> Dict:
    dataset = _get_time_series_dataset(ctx, ds_name, var_name)
    if not GeoJSON.is_geometry(geometry):
        raise ServiceBadRequestError("Invalid GeoJSON geometry")
    if isinstance(geometry, dict):
        geometry = shapely.geometry.shape(geometry)
    return _get_time_series_for_geometry(dataset, var_name,
                                         geometry,
                                         start_date=start_date, end_date=end_date)


def get_time_series_for_geometry_collection(ctx: ServiceContext,
                                            ds_name: str, var_name: str,
                                            geometry_collection: Dict,
                                            start_date: np.datetime64 = None,
                                            end_date: np.datetime64 = None) -> Dict:
    dataset = _get_time_series_dataset(ctx, ds_name, var_name)
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
    return _get_time_series_for_geometries(dataset, var_name, shapes, start_date, end_date)


def get_time_series_for_feature_collection(ctx: ServiceContext,
                                           ds_name: str, var_name: str,
                                           feature_collection: Dict,
                                           start_date: np.datetime64 = None,
                                           end_date: np.datetime64 = None) -> Dict:
    dataset = _get_time_series_dataset(ctx, ds_name, var_name)
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
    return _get_time_series_for_geometries(dataset, var_name, shapes, start_date, end_date)


def _get_time_series_for_point(dataset: xr.Dataset,
                               var_name: str,
                               point: shapely.geometry.Point,
                               start_date: np.datetime64 = None,
                               end_date: np.datetime64 = None) -> Dict:
    var_names = [var_name]

    ancillary_var_names = find_ancillary_var_names(dataset, var_name, same_shape=True, same_dims=True)
    uncert_var_name = None
    if 'standard_error' in ancillary_var_names:
        uncert_var_name = next(iter(ancillary_var_names['standard_error']))
        var_names.append(uncert_var_name)

    ts_ds = ts.get_time_series(dataset, point, var_names, start_date=start_date, end_date=end_date)
    if ts_ds is None:
        return {'results': []}

    return _collect_ts_result(ts_ds, var_name, uncert_var_name, None)


def _get_time_series_for_geometry(dataset: xr.Dataset,
                                  var_name: str,
                                  geometry: shapely.geometry.base.BaseGeometry,
                                  start_date: np.datetime64 = None,
                                  end_date: np.datetime64 = None) -> Dict:
    if isinstance(geometry, shapely.geometry.Point):
        return _get_time_series_for_point(dataset, var_name,
                                          geometry,
                                          start_date=start_date, end_date=end_date)

    ts_ds = ts.get_time_series(dataset, geometry, [var_name],
                               include_count=True,
                               include_stdev=False,
                               start_date=start_date, end_date=end_date)
    if ts_ds is None:
        return {'results': []}

    ancillary_var_names = find_ancillary_var_names(ts_ds, var_name)

    uncert_var_name = None
    if 'standard_error' in ancillary_var_names:
        uncert_var_name = next(iter(ancillary_var_names['standard_error']))

    count_var_name = None
    if 'number_of_observations' in ancillary_var_names:
        count_var_name = next(iter(ancillary_var_names['number_of_observations']))

    return _collect_ts_result(ts_ds, var_name, uncert_var_name, count_var_name)


def _collect_ts_result(ts_ds, var_name, uncert_var_name, count_var_name):
    var = ts_ds[var_name]
    uncert_var = ts_ds[uncert_var_name] if uncert_var_name else None
    count_var = ts_ds[count_var_name] if count_var_name else None

    total_count = ts_ds.get('max_number_of_observations', 1)

    num_times = var.time.size
    time_series = []
    for time_index in range(num_times):
        value = float(var[time_index])
        statistics = {'totalCount': total_count}
        if np.isnan(value):
            statistics['average'] = None
            statistics['validCount'] = 0
        else:
            statistics['average'] = value
            statistics['validCount'] = int(count_var[time_index]) if count_var is not None else 1
        if uncert_var is not None:
            value = float(uncert_var[time_index])
            # TODO (forman): agree with Dirk on how we call provided uncertainty
            if np.isnan(value):
                statistics['uncertainty'] = None
            else:
                statistics['uncertainty'] = float(value)

        time_series.append(dict(result=statistics,
                                date=timestamp_to_iso_string(var.time[time_index].data)))

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


def _get_time_series_dataset(ctx: ServiceContext, ds_name: str, var_name: str = None):
    descriptor = ctx.get_dataset_descriptor(ds_name)
    ts_ds_name = descriptor.get('TimeSeriesDataset', ds_name)
    return ctx.get_dataset(ts_ds_name, expected_var_names=[var_name] if var_name else None)
