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

"""
This module implements the controller for the "/ts" handler.
It is maintained for legacy reasons only (DCS4COP VITO viewer).
"""

from typing import Dict, List, Optional

import numpy as np
import shapely.geometry
import xarray as xr

from xcube.constants import LOG
from xcube.core import timeseries
from xcube.core.ancvar import find_ancillary_var_names
from xcube.core.geom import get_dataset_bounds
from xcube.core.timecoord import timestamp_to_iso_string
from xcube.util.geojson import GeoJSON
from xcube.util.perf import measure_time
from xcube.webapi.context import ServiceContext
from xcube.webapi.errors import ServiceBadRequestError


def get_time_series_info(ctx: ServiceContext) -> Dict:
    """
    Get time-series meta-information for variables.

    :param ctx: Service context object
    :return: a dictionary with a single entry "layers" which is a list of entries that are
             dictionaries containing a variable's "name", "dates", and "bounds".
    """
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
                              end_date: np.datetime64 = None,
                              max_valids: int = None) -> Dict:
    """
    Get the time-series for a given point.

    :param ctx: Service context object
    :param ds_name: The dataset identifier.
    :param var_name: The variable name.
    :param lon: The point's longitude in decimal degrees.
    :param lat: The point's latitude in decimal degrees.
    :param start_date: An optional start date.
    :param end_date: An optional end date.
    :param max_valids: Optional number of valid points.
           If it is None (default), also missing values are returned as NaN;
           if it is -1 only valid values are returned;
           if it is a positive integer, the most recent valid values are returned.
    :return: Time-series data structure.
    """
    dataset = ctx.get_time_series_dataset(ds_name, var_name=var_name)
    with measure_time() as time_result:
        result = _get_time_series_for_point(dataset, var_name,
                                            shapely.geometry.Point(lon, lat),
                                            start_date=start_date,
                                            end_date=end_date,
                                            max_valids=max_valids)
    if ctx.trace_perf:
        LOG.info(f'get_time_series_for_point:: dataset id {ds_name}, variable {var_name}, '
                 f'geometry type {shapely.geometry.Point(lon, lat)}, size={len(result["results"])}, '
                 f'took {time_result.duration} seconds')
    return result


def get_time_series_for_geometry(ctx: ServiceContext,
                                 ds_name: str, var_name: str,
                                 geometry: Dict,
                                 start_date: np.datetime64 = None,
                                 end_date: np.datetime64 = None,
                                 include_count: bool = False,
                                 include_stdev: bool = False,
                                 max_valids: int = None) -> Dict:
    """
    Get the time-series for a given *geometry*.

    :param ctx: Service context object
    :param ds_name: The dataset identifier.
    :param var_name: The variable name.
    :param geometry: The geometry, usually a point or polygon.
    :param start_date: An optional start date.
    :param end_date: An optional end date.
    :param include_count: Whether to include the valid number of observations in the result.
    :param include_stdev: Whether to include the standard deviation in the result.
    :param max_valids: Optional number of valid points.
           If it is None (default), also missing values are returned as NaN;
           if it is -1 only valid values are returned;
           if it is a positive integer, the most recent valid values are returned.
    :return: Time-series data structure.
    """
    dataset = ctx.get_time_series_dataset(ds_name, var_name=var_name)
    if not GeoJSON.is_geometry(geometry):
        raise ServiceBadRequestError("Invalid GeoJSON geometry")
    if isinstance(geometry, dict):
        geometry = shapely.geometry.shape(geometry)
    with measure_time() as time_result:
        result = _get_time_series_for_geometry(dataset, var_name,
                                               geometry,
                                               start_date=start_date,
                                               end_date=end_date,
                                               include_count=include_count,
                                               include_stdev=include_stdev,
                                               max_valids=max_valids)

    if ctx.trace_perf:
        LOG.info(f'get_time_series_for_geometry: dataset id {ds_name}, variable {var_name}, '
                 f'geometry type {geometry},'
                 f'size={len(result["results"])}, took {time_result.duration} seconds')
    return result


def get_time_series_for_geometry_collection(ctx: ServiceContext,
                                            ds_name: str, var_name: str,
                                            geometry_collection: Dict,
                                            start_date: np.datetime64 = None,
                                            end_date: np.datetime64 = None,
                                            include_count: bool = False,
                                            include_stdev: bool = False,
                                            max_valids: int = None) -> Dict:
    """
    Get the time-series for a given *geometry_collection*.

    :param ctx: Service context object
    :param ds_name: The dataset identifier.
    :param var_name: The variable name.
    :param geometry_collection: The geometry collection.
    :param start_date: An optional start date.
    :param end_date: An optional end date.
    :param include_count: Whether to include the valid number of observations in the result.
    :param include_stdev: Whether to include the standard deviation in the result.
    :param max_valids: Optional number of valid points.
           If it is None (default), also missing values are returned as NaN;
           if it is -1 only valid values are returned;
           if it is a positive integer, the most recent valid values are returned.
    :return: Time-series data structure.
    """
    dataset = ctx.get_time_series_dataset(ds_name, var_name=var_name)
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
    with measure_time() as time_result:
        result = _get_time_series_for_geometries(dataset, var_name, shapes,
                                                 start_date=start_date,
                                                 end_date=end_date,
                                                 include_count=include_count,
                                                 include_stdev=include_stdev,
                                                 max_valids=max_valids)
    if ctx.trace_perf:
        LOG.info(f'get_time_series_for_geometry_collection: dataset id {ds_name}, variable {var_name},'
                 f'size={len(result["results"])}, took {time_result.duration} seconds')
    return result


def get_time_series_for_feature_collection(ctx: ServiceContext,
                                           ds_name: str, var_name: str,
                                           feature_collection: Dict,
                                           start_date: np.datetime64 = None,
                                           end_date: np.datetime64 = None,
                                           include_count: bool = False,
                                           include_stdev: bool = False,
                                           max_valids: int = None) -> Dict:
    """
    Get the time-series for the geometries of a given *feature_collection*.

    :param ctx: Service context object
    :param ds_name: The dataset identifier.
    :param var_name: The variable name.
    :param feature_collection: The feature collection.
    :param start_date: An optional start date.
    :param end_date: An optional end date.
    :param include_count: Whether to include the valid number of observations in the result.
    :param include_stdev: Whether to include the standard deviation in the result.
    :param max_valids: Optional number of valid points.
           If it is None (default), also missing values are returned as NaN;
           if it is -1 only valid values are returned;
           if it is a positive integer, the most recent valid values are returned.
    :return: Time-series data structure.
    """
    dataset = ctx.get_time_series_dataset(ds_name, var_name=var_name)
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
    with measure_time() as time_result:
        result = _get_time_series_for_geometries(dataset, var_name, shapes,
                                                 start_date=start_date,
                                                 end_date=end_date,
                                                 include_count=include_count,
                                                 include_stdev=include_stdev,
                                                 max_valids=max_valids)
    if ctx.trace_perf:
        LOG.info(f'get_time_series_for_feature_collection: dataset id {ds_name}, variable {var_name},'
                 f'size={len(result["results"])}, took {time_result.duration} seconds')
    return result


def _get_time_series_for_point(dataset: xr.Dataset,
                               var_name: str,
                               point: shapely.geometry.Point,
                               start_date: np.datetime64 = None,
                               end_date: np.datetime64 = None,
                               max_valids: int = None) -> Dict:
    var_names = [var_name]

    ancillary_var_names = find_ancillary_var_names(dataset, var_name, same_shape=True, same_dims=True)
    uncert_var_name = None
    if 'standard_error' in ancillary_var_names:
        uncert_var_name = next(iter(ancillary_var_names['standard_error']))
        var_names.append(uncert_var_name)

    ts_ds = timeseries.get_time_series(dataset, point, var_names, start_date=start_date, end_date=end_date)
    if ts_ds is None:
        return {'results': []}

    return _collect_ts_result(ts_ds,
                              var_name,
                              uncert_var_name=uncert_var_name,
                              count_var_name=None,
                              max_valids=max_valids)


def _get_time_series_for_geometry(dataset: xr.Dataset,
                                  var_name: str,
                                  geometry: shapely.geometry.base.BaseGeometry,
                                  start_date: np.datetime64 = None,
                                  end_date: np.datetime64 = None,
                                  include_count=True,
                                  include_stdev=False,
                                  max_valids: int = None) -> Dict:
    if isinstance(geometry, shapely.geometry.Point):
        return _get_time_series_for_point(dataset, var_name,
                                          geometry,
                                          start_date=start_date, end_date=end_date)

    agg_methods = set()
    agg_methods.add('mean')
    if include_stdev:
        agg_methods.add('std')
    if include_count:
        agg_methods.add('count')

    ts_ds = timeseries.get_time_series(dataset,
                                       geometry,
                                       [var_name],
                                       agg_methods=agg_methods,
                                       start_date=start_date,
                                       end_date=end_date)
    if ts_ds is None:
        return {'results': []}

    ancillary_var_names = find_ancillary_var_names(ts_ds, var_name)

    uncert_var_name = None
    if 'standard_error' in ancillary_var_names:
        uncert_var_name = next(iter(ancillary_var_names['standard_error']))

    count_var_name = None
    if 'number_of_observations' in ancillary_var_names:
        count_var_name = next(iter(ancillary_var_names['number_of_observations']))

    return _collect_ts_result(ts_ds,
                              var_name,
                              uncert_var_name=uncert_var_name,
                              count_var_name=count_var_name,
                              max_valids=max_valids)


def _collect_ts_result(ts_ds: xr.Dataset,
                       var_name: str,
                       uncert_var_name: str = None,
                       count_var_name: str = None,
                       max_valids: int = None):
    if not (max_valids is None or max_valids == -1 or max_valids > 0):
        raise ValueError('max_valids must be either None, -1 or positive')

    average_var = ts_ds.get(var_name, ts_ds.get(var_name + '_mean'))
    uncert_var = ts_ds.get(uncert_var_name) if uncert_var_name else None
    count_var = ts_ds.get(count_var_name) if count_var_name else None

    total_count_value = ts_ds.attrs.get('max_number_of_observations', 1)

    num_times = average_var.time.size
    time_series = []

    pos_max_valids = max_valids is not None and max_valids > 0
    if pos_max_valids:
        time_indexes = range(num_times - 1, -1, -1)
    else:
        time_indexes = range(num_times)

    average_values = average_var.values
    count_values = count_var.values if count_var is not None else None
    uncert_values = uncert_var.values if uncert_var is not None else None

    for time_index in time_indexes:
        if len(time_series) == max_valids:
            break

        average_value = _get_float_value(average_values, time_index)
        if average_value is None:
            if max_valids is not None:
                continue
            count_value = 0
        else:
            count_value = int(count_values[time_index]) if count_values is not None else 1

        statistics = {
            'average': average_value,
            'validCount': count_value,
            'totalCount': total_count_value
        }
        if uncert_values is not None:
            statistics['uncertainty'] = _get_float_value(uncert_values, time_index)

        time_series.append(dict(result=statistics,
                                date=timestamp_to_iso_string(average_var.time[time_index].values)))

    if pos_max_valids:
        return {'results': time_series[::-1]}
    else:
        return {'results': time_series}


def _get_float_value(values: Optional[np.ndarray], index: int) -> Optional[float]:
    if values is None:
        return None
    value = float(values[index])
    return None if np.isnan(value) else value


def _get_time_series_for_geometries(dataset: xr.Dataset,
                                    var_name: str,
                                    geometries: List[shapely.geometry.base.BaseGeometry],
                                    start_date: np.datetime64 = None,
                                    end_date: np.datetime64 = None,
                                    include_count=False,
                                    include_stdev=False,
                                    max_valids=None) -> Dict:
    time_series = []
    for geometry in geometries:
        result = _get_time_series_for_geometry(dataset, var_name,
                                               geometry,
                                               start_date=start_date,
                                               end_date=end_date,
                                               include_count=include_count,
                                               include_stdev=include_stdev,
                                               max_valids=max_valids)
        time_series.append(result["results"])
    return {'results': time_series}
