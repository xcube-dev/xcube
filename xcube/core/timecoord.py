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

import datetime
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

REF_DATETIME_STR = '1970-01-01 00:00:00'
REF_DATETIME = pd.to_datetime(REF_DATETIME_STR, utc=True)
DATETIME_UNITS = f'days since {REF_DATETIME_STR}'
DATETIME_CALENDAR = 'gregorian'
SECONDS_PER_DAY = 24 * 60 * 60
MICROSECONDS_PER_DAY = 1000 * 1000 * SECONDS_PER_DAY


def add_time_coords(dataset: xr.Dataset, time_range: Tuple[float, float]) -> xr.Dataset:
    t1, t2 = time_range
    if t1 != t2:
        t_center = (t1 + t2) / 2
    else:
        t_center = t1
    dataset = dataset.expand_dims('time')
    dataset = dataset.assign_coords(time=(['time'],
                                          from_time_in_days_since_1970([t_center])))
    time_var = dataset.coords['time']
    time_var.attrs['long_name'] = 'time'
    time_var.attrs['standard_name'] = 'time'
    # Avoiding xarray error:
    #   ValueError: failed to prevent overwriting existing key units in attrs on variable 'time'.
    #   This is probably an encoding field used by xarray to describe how a variable is serialized.
    #   To proceed, remove this key from the variable's attributes manually.
    # time_var.attrs['units'] = DATETIME_UNITS
    # time_var.attrs['calendar'] = DATETIME_CALENDAR
    time_var.encoding['units'] = DATETIME_UNITS
    time_var.encoding['calendar'] = DATETIME_CALENDAR
    if t1 != t2:
        time_var.attrs['bounds'] = 'time_bnds'
        dataset = dataset.assign_coords(time_bnds=(['time', 'bnds'],
                                                   from_time_in_days_since_1970([t1, t2]).reshape(1, 2)))
        time_bnds_var = dataset.coords['time_bnds']
        time_bnds_var.attrs['long_name'] = 'time'
        time_bnds_var.attrs['standard_name'] = 'time'
        # Avoiding xarray error:
        #   ValueError: failed to prevent overwriting existing key units in attrs on variable 'time'.
        #   This is probably an encoding field used by xarray to describe how a variable is serialized.
        #   To proceed, remove this key from the variable's attributes manually.
        # time_bnds_var.attrs['units'] = DATETIME_UNITS
        # time_bnds_var.attrs['calendar'] = DATETIME_CALENDAR
        time_bnds_var.encoding['units'] = DATETIME_UNITS
        time_bnds_var.encoding['calendar'] = DATETIME_CALENDAR
    return dataset


def get_time_range_from_data(dataset: xr.Dataset, maybe_consider_metadata: bool=True) \
        -> Tuple[Optional[float], Optional[float]]:
    """
    Determines a time range from a dataset by inspecting its time_bounds or time data arrays.
    In cases where no time bounds are given and no time periodicity can be determined,
    metadata may be considered.

    :param dataset: The dataset of which the time range shall be determined
    "param maybe_consider_metadata": Whether metadata shall be considered.
    Only used when the dataset has no time bounds array and no time periodicity.
    The values will only be set when they do not contradict the values from the data arrays.
    :return: A tuple with two float values: The first one represents the start time,
    the second the end time. Either may be None.
    """
    time_bounds_names = ['time_bnds', 'time_bounds']
    for time_bounds_name in time_bounds_names:
        if time_bounds_name in dataset:
            return _get_time_range_from_time_bounds(dataset, time_bounds_name)
    if 'start_time' in dataset and 'end_time' in dataset:
        return dataset['start_time'].values[0], dataset['end_time'].values[-1]
    time_names = ['time', 't']
    time = None
    for time_name in time_names:
        if time_name in dataset:
            time = dataset[time_name]
    if time is None:
        return None, None
    time_bnds_name = time.attrs.get("bounds", "time_bnds")
    if time_bnds_name in dataset:
        return _get_time_range_from_time_bounds(dataset, time_bnds_name)
    if time.size == 1:
        return _maybe_return_time_range_from_metadata(dataset,
                                                      time.values[0],
                                                      time.values[0],
                                                      maybe_consider_metadata)
    if time.size == 2:
        return _maybe_return_time_range_from_metadata(dataset,
                                                      time.values[0],
                                                      time.values[1],
                                                      maybe_consider_metadata)
    time_diff = time.diff(dim=time.dims[0]).values
    time_res = time_diff[0]
    time_regular = all([time_res - diff == np.timedelta64(0) for diff in time_diff[1:]])
    if time_regular:
        try:
            return time.values[0] - time_res / 2, time.values[-1] + time_res / 2
        except TypeError:
            # Time is probably given as cftime.DatetimeJulian or cftime.DatetimeGregorian
            # To convert it to datetime, we must derive its isoformat first.
            return (pd.to_datetime(time.values[0].isoformat()) - time_res / 2).to_datetime64(), \
                   (pd.to_datetime(time.values[-1].isoformat()) + time_res / 2).to_datetime64()
    return _maybe_return_time_range_from_metadata(dataset,
                                                  time.values[0],
                                                  time.values[-1],
                                                  maybe_consider_metadata)


def _maybe_return_time_range_from_metadata(dataset: xr.Dataset,
                                           data_start_time: float,
                                           data_end_time: float,
                                           maybe_consider_metadata: bool) -> Tuple[float, float]:
    if maybe_consider_metadata:
        attr_start_time, attr_end_time = get_time_range_from_attrs(dataset)
        attr_start_time = pd.to_datetime(attr_start_time, infer_datetime_format=False, utc=True)
        attr_end_time = pd.to_datetime(attr_end_time, infer_datetime_format=False, utc=True)
        if attr_start_time is not None and attr_end_time is not None:
            try:
                if attr_start_time < data_start_time and attr_end_time > data_end_time:
                    return attr_start_time.to_datetime64(), attr_end_time.to_datetime64()
            except TypeError:
                try:
                    if attr_start_time.to_datetime64() < data_start_time \
                            and attr_end_time.to_datetime64() > data_end_time:
                        return attr_start_time.to_datetime64(), attr_end_time.to_datetime64()
                except TypeError:
                    # use time values from data
                    pass
    return data_start_time, data_end_time


def _get_time_range_from_time_bounds(dataset: xr.Dataset, time_bounds_name: str) \
        -> Tuple[Optional[float], Optional[float]]:
    time_bnds = dataset[time_bounds_name]
    if len(time_bnds.shape) == 2 and time_bnds.shape[1] == 2:
        return min(time_bnds[:, 0]).values, max(time_bnds[:, 1]).values


def get_time_range_from_attrs(dataset: xr.Dataset) -> Tuple[Optional[str], Optional[str]]:
    return get_start_time_from_attrs(dataset), get_end_time_from_attrs(dataset)


def get_start_time_from_attrs(dataset: xr.Dataset) -> Optional[str]:
    return _get_attr(dataset, ['time_coverage_start', 'time_start', 'start_time', 'start_date'])


def get_end_time_from_attrs(dataset: xr.Dataset) -> Optional[str]:
    return _get_attr(dataset, ['time_coverage_end', 'time_stop', 'time_end', 'stop_time',
                               'end_time', 'stop_date', 'end_date'])


def _get_attr(dataset: xr.Dataset, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in dataset.attrs:
            return remove_time_part_from_isoformat(str(dataset.attrs[name]))


def remove_time_part_from_isoformat(datetime_str: str) -> str:
    date_length = 10  # for example len("2010-02-04") == 10
    if len(datetime_str) > date_length and datetime_str[date_length] in ('T', ' '):
        return datetime_str[0: date_length]
    return datetime_str


def to_time_in_days_since_1970(time_str: str, pattern=None) -> float:
    datetime = pd.to_datetime(time_str, format=pattern, infer_datetime_format=False, utc=True)
    timedelta = datetime - REF_DATETIME
    return timedelta.days + timedelta.seconds / SECONDS_PER_DAY + \
           timedelta.microseconds / MICROSECONDS_PER_DAY


def from_time_in_days_since_1970(time_value: Union[float, Sequence[float]]) -> np.ndarray:
    if isinstance(time_value, int) or isinstance(time_value, float):
        return pd.to_datetime(time_value, utc=True, unit='d', origin='unix').round(freq='ms').to_datetime64()
    else:
        return np.array(list(map(from_time_in_days_since_1970, time_value)))


def timestamp_to_iso_string(time: Union[np.datetime64, datetime.datetime], freq='S'):
    """
    Convert a UTC timestamp given as nanos, millis, seconds, etc. since 1970-01-01 00:00:00
    to an ISO-format string.

    :param time: UTC timestamp given as time delta since since 1970-01-01 00:00:00 in the units given by
           the numpy datetime64 type, so it can be as nanos, millis, seconds since 1970-01-01 00:00:00.
    :param freq: time rounding resolution. See pandas.Timestamp.round().
    :return: ISO-format string.
    """
    # All times are UTC (Z = Zulu Time Zone = UTC)
    return pd.Timestamp(time).round(freq).isoformat() + 'Z'
