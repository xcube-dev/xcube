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

from typing import Dict, Any, Sequence, Union, List

import cftime
import numpy as np
import pandas as pd
import re
import xarray as xr

from xcube.core.schema import CubeSchema
from xcube.core.select import select_variables_subset
from xcube.core.verify import assert_cube

UPSAMPLING_METHODS = ['asfreq', 'ffill', 'bfill', 'pad', 'nearest',
                      'interpolate']
DOWNSAMPLING_METHODS = ['count', 'first', 'last', 'min', 'max', 'sum', 'prod',
                        'mean', 'median', 'std', 'var', 'percentile']
SPLINE_INTERPOLATION_KINDS = ['zero', 'slinear', 'quadratic', 'cubic']
OTHER_INTERPOLATION_KINDS = ['linear', 'nearest', 'previous', 'next']
INTERPOLATION_KINDS = SPLINE_INTERPOLATION_KINDS + OTHER_INTERPOLATION_KINDS

TIMEUNIT_INCREMENTS = dict(
    YS=[1, 0, 0, 0],
    QS=[0, 3, 0, 0],
    MS=[0, 1, 0, 0]
)
HALF_TIMEUNIT_INCREMENTS = dict(
    YS=[0, 6, 0, 0]
)


def resample_in_time(dataset: xr.Dataset,
                     frequency: str,
                     method: Union[str, Sequence[str]],
                     offset=None,
                     base: int = 0,
                     tolerance=None,
                     interp_kind=None,
                     time_chunk_size=None,
                     var_names: Sequence[str] = None,
                     metadata: Dict[str, Any] = None,
                     cube_asserted: bool = False,
                     rename_variables: bool = True) -> xr.Dataset:
    """
    Resample a dataset in the time dimension.

    The argument *method* may be one or a sequence of
    ``'all'``, ``'any'``,
    ``'argmax'``, ``'argmin'``, ``'count'``,
    ``'first'``, ``'last'``,
    ``'max'``, ``'min'``, ``'mean'``, ``'median'``,
    ``'percentile_<p>'``,
    ``'std'``, ``'sum'``, ``'var'``,
    ``'interpolate'``
    .

    In value ``'percentile_<p>'`` is a placeholder,
    where ``'<p>'`` must be replaced by an integer percentage
    value, e.g. ``'percentile_90'`` is the 90%-percentile.

    *Important note:* As of xarray 0.14 and dask 2.8, the
    methods ``'median'`` and ``'percentile_<p>'` cannot be
    used if the variables in *cube* comprise chunked dask arrays.
    In this case, use the ``compute()`` or ``load()`` method
    to convert dask arrays into numpy arrays.

    :param dataset: The xcube dataset.
    :param frequency: Temporal aggregation frequency.
        Use format "<count><offset>" where <offset> is one of
        'H', 'D', 'W', 'M', 'Q', 'Y'.
    :param method: Resampling method or sequence of
        resampling methods.
    :param offset: Offset used to adjust the resampled time labels.
        Uses same syntax as *frequency*.
    :param base: For frequencies that evenly subdivide 1 day,
        the "origin" of the aggregated intervals. For example,
        for '24H' frequency, base could range from 0 through 23.
    :param time_chunk_size: If not None, the chunk size to be
        used for the "time" dimension.
    :param var_names: Variable names to include.
    :param tolerance: Time tolerance for selective
        upsampling methods. Defaults to *frequency*.
    :param interp_kind: Kind of interpolation
        if *method* is 'interpolation'.
    :param metadata: Output metadata.
    :param cube_asserted: If False, *cube* will be verified,
        otherwise it is expected to be a valid cube.
    :param rename_variables: Whether the dataset's variables shall be renamed by
        extending the resampling method to the original name.
    :return: A new xcube dataset resampled in time.
    """
    if not cube_asserted:
        assert_cube(dataset)

    if frequency == 'all':
        time_gap = np.array(dataset.time[-1]) - np.array(dataset.time[0])
        days = int((np.timedelta64(time_gap, 'D')
                    / np.timedelta64(1, 'D')) + 1)
        frequency = f'{days}D'

    # resample to start of period
    if frequency.endswith('Y') or frequency.endswith('M') or \
            frequency.endswith('Q'):
        frequency = f'{frequency}S'

    if var_names:
        dataset = select_variables_subset(dataset, var_names)

    resampler = dataset.resample(skipna=True,
                                 closed='left',
                                 label='left',
                                 time=frequency,
                                 loffset=offset,
                                 base=base)

    if isinstance(method, str):
        methods = [method]
    else:
        methods = list(method)

    percentile_prefix = 'percentile_'

    resampled_cubes = []
    for method in methods:
        method_args = []
        method_postfix = method
        if method.startswith(percentile_prefix):
            p = int(method[len(percentile_prefix):])
            q = p / 100.0
            method_args = [q]
            method_postfix = f'p{p}'
            method = 'quantile'
        resampling_method = getattr(resampler, method)
        method_kwargs = get_method_kwargs(method,
                                          frequency,
                                          interp_kind,
                                          tolerance)
        resampled_cube = resampling_method(*method_args,
                                           **method_kwargs)
        if rename_variables:
            resampled_cube = resampled_cube.rename(
                {var_name: f'{var_name}_{method_postfix}'
                 for var_name in resampled_cube.data_vars})
        resampled_cubes.append(resampled_cube)

    if len(resampled_cubes) == 1:
        resampled_cube = resampled_cubes[0]
    else:
        resampled_cube = xr.merge(resampled_cubes)
    adjusted_times, time_bounds = _adjust_times_and_bounds(
        resampled_cube.time.values, frequency, method)
    update_vars = dict(
        time=adjusted_times,
        time_bnds=xr.DataArray(time_bounds, dims=['time', 'bnds'])
    )
    resampled_cube = resampled_cube.assign_coords(update_vars)

    return adjust_metadata_and_chunking(resampled_cube,
                                        metadata=metadata,
                                        time_chunk_size=time_chunk_size)


def adjust_metadata_and_chunking(dataset, metadata=None, time_chunk_size=None):
    time_coverage_start = '%s' % dataset.time_bnds[0][0]
    time_coverage_end = '%s' % dataset.time_bnds[-1][1]

    dataset.attrs.update(metadata or {})
    # TODO: add other time_coverage_ attributes
    dataset.attrs.update(time_coverage_start=time_coverage_start,
                         time_coverage_end=time_coverage_end)
    try:
        schema = CubeSchema.new(dataset)
    except ValueError:
        return _adjust_chunk_sizes_without_schema(dataset, time_chunk_size)
    if schema.chunks is None:
        return _adjust_chunk_sizes_without_schema(dataset, time_chunk_size)

    chunk_sizes = {schema.dims[i]: schema.chunks[i] for i in range(schema.ndim)}

    if isinstance(time_chunk_size, int) and time_chunk_size >= 0:
        chunk_sizes['time'] = time_chunk_size

    return dataset.chunk(chunk_sizes)


def _adjust_chunk_sizes_without_schema(dataset, time_chunk_size=None):
    chunk_sizes = dict(dataset.chunks)
    if isinstance(time_chunk_size, int) and time_chunk_size >= 0:
        chunk_sizes['time'] = time_chunk_size
    else:
        chunk_sizes['time'] = 1
    return dataset.chunk(chunk_sizes)


def _adjust_times_and_bounds(time_values, frequency, method):
    time_unit = re.findall('[A-Z]+', frequency)[0]
    time_value = int(frequency.split(time_unit)[0])
    if time_unit not in TIMEUNIT_INCREMENTS:
        if time_unit == 'D':
            half_time_delta = np.timedelta64(12 * time_value, 'h')
        elif time_unit == 'H':
            half_time_delta = np.timedelta64(30 * time_value, 'm')
        elif time_unit == 'W':
            half_time_delta = np.timedelta64(84 * time_value, 'h')
        else:
            raise ValueError(f'Unsupported time unit "{time_unit}"')
        if method not in UPSAMPLING_METHODS:
            time_values += half_time_delta
        time_bounds_values = \
            np.array([time_values - half_time_delta,
                      time_values + half_time_delta]).transpose()
        return time_values, time_bounds_values
    # time units year, month and quarter cannot be converted to
    # numpy timedelta objects, so we have to convert them to pandas timestamps
    # and modify these
    is_cf_time = isinstance(time_values[0], cftime.datetime)
    if is_cf_time:
        timestamps = [pd.Timestamp(tv.isoformat()) for tv in time_values]
        calendar = time_values[0].calendar
    else:
        timestamps = [pd.Timestamp(tv) for tv in time_values]
        calendar = None

    timestamps.append(_get_next_timestamp(timestamps[-1],
                                          time_unit,
                                          time_value,
                                          False))

    new_timestamps = []
    new_timestamp_bounds = []
    for i, ts in enumerate(timestamps[:-1]):
        next_ts = timestamps[i + 1]
        half_next_ts = _get_next_timestamp(ts, time_unit, time_value, True)
        # depending on whether the data was sampled down or up,
        # times need to be adjusted differently
        if method not in UPSAMPLING_METHODS:
            new_timestamps.append(_convert(half_next_ts, calendar))
            new_timestamp_bounds.append([_convert(ts, calendar),
                                         _convert(next_ts, calendar)])
        else:
            half_previous_ts = \
                _get_previous_timestamp(ts, time_unit, time_value, True)
            new_timestamps.append(_convert(ts, calendar))
            new_timestamp_bounds.append([_convert(half_previous_ts,
                                                  calendar),
                                         _convert(half_next_ts,
                                                  calendar)])
    return new_timestamps, new_timestamp_bounds


def _convert(timestamp: pd.Timestamp, calendar: str):
    if calendar is not None:
        return cftime.datetime.fromordinal(timestamp.to_julian_date(),
                                           calendar=calendar)
    return np.datetime64(timestamp)


def _get_next_timestamp(timestamp, time_unit, time_value, half) \
        -> pd.Timestamp:
    # Retrieves the timestamp following the passed timestamp according to the
    # given time unit and time value.
    # If half is True, the timestamp halfway between the timestamp and the next
    # timestamp (which is not necessarily halfway between the two) is returned
    increments = _get_increments(timestamp, time_unit, time_value, half)
    replacement = dict(
        year=timestamp.year + increments[0],
        month=timestamp.month + increments[1],
        day=timestamp.day + increments[2],
        hour=timestamp.hour + increments[3]
    )
    while replacement['hour'] > 24:
        replacement['hour'] -= 24
        replacement['day'] += 1
    while replacement['day'] > _days_of_month(replacement['year'],
                                              replacement['month']):
        replacement['day'] -= _days_of_month(replacement['year'],
                                             replacement['month'])
        replacement['month'] += 1
        if replacement['month'] > 12:
            replacement['month'] -= 12
            replacement['year'] += 1

    while replacement['month'] > 12:
        replacement['month'] -= 12
        replacement['year'] += 1

    return pd.Timestamp(timestamp.replace(**replacement))


def _get_previous_timestamp(timestamp, time_unit, time_value, half) \
        -> pd.Timestamp:
    # Retrieves the timestamp preceding the passed timestamp according to the
    # given time unit and time value.
    # If half is True, the timestamp halfway between the timestamp and the
    # previous timestamp (which is not necessarily halfway between the two)
    # is returned
    increments = _get_increments(timestamp, time_unit, time_value, half)
    replacement = dict(
        year=timestamp.year - increments[0],
        month=timestamp.month - increments[1],
        day=timestamp.day - increments[2],
        hour=timestamp.hour - increments[3]
    )

    while replacement['hour'] < 0:
        replacement['hour'] += 24
        replacement['day'] -= 1

    while replacement['day'] < 1:
        replacement['month'] -= 1
        if replacement['month'] < 1:
            replacement['month'] += 12
            replacement['year'] -= 1
        replacement['day'] += _days_of_month(replacement['year'],
                                             replacement['month'] % 12)

    while replacement['month'] < 1:
        replacement['month'] += 12
        replacement['year'] -= 1

    return pd.Timestamp(timestamp.replace(**replacement))


def _get_increments(timestamp, time_unit, time_value, half) -> List[int]:
    # Determines the increments for year, month, day, and hour to be applied
    # to a timestamp
    if not half:
        return _tune_increments(TIMEUNIT_INCREMENTS[time_unit], time_value)
    if time_value % 2 == 0:
        time_value /= 2
        return _tune_increments(TIMEUNIT_INCREMENTS[time_unit],
                                int(time_value))
    if time_unit in HALF_TIMEUNIT_INCREMENTS:
        return _tune_increments(HALF_TIMEUNIT_INCREMENTS[time_unit],
                                time_value)
    if time_unit == 'QS':
        num_months = 3
    else:
        num_months = 1
    import math
    month = int(math.floor((num_months * time_value) / 2))
    days = _days_of_month(timestamp.year, month)
    if days % 2 == 0:
        hours = 0
    else:
        hours = 12
    days = int(math.floor(days / 2)) - 1
    return [0, month, days, hours]


def _tune_increments(incrementors, time_value):
    incrementors = [i * time_value for i in incrementors]
    return incrementors


def _days_of_month(year: int, month: int):
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    if month in [4, 6, 9, 11]:
        return 30
    if year % 4 != 0:
        return 28
    if year % 400 == 0:
        return 29
    if year % 100 == 0:
        return 28
    return 28


def get_method_kwargs(method, frequency, interp_kind, tolerance):
    if method == 'interpolate':
        kwargs = {'kind': interp_kind or 'linear'}
    elif method in {'nearest', 'bfill', 'ffill', 'pad'}:
        kwargs = {'tolerance': tolerance or frequency}
    elif method in {'last', 'sum',
                    'min', 'max',
                    'mean', 'median', 'std', 'var'}:
        kwargs = {'dim': 'time', 'keep_attrs': True, 'skipna': True}
    elif method == 'first':
        kwargs = {'keep_attrs': True, 'skipna': False}
    elif method == 'prod':
        kwargs = {'dim': 'time', 'skipna': True}
    elif method == 'count':
        kwargs = {'dim': 'time', 'keep_attrs': True}
    else:
        kwargs = {}
    return kwargs
