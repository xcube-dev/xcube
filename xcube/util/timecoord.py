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
from typing import Tuple, Union, Sequence

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
    dataset = dataset.assign_coords(time=(['time'], [t_center]))
    time_var = dataset.coords['time']
    time_var.attrs['long_name'] = 'time'
    time_var.attrs['standard_name'] = 'time'
    time_var.attrs['units'] = DATETIME_UNITS
    time_var.attrs['calendar'] = DATETIME_CALENDAR
    time_var.encoding['units'] = DATETIME_UNITS
    time_var.encoding['calendar'] = DATETIME_CALENDAR
    if t1 != t2:
        time_var.attrs['bounds'] = 'time_bnds'
        dataset = dataset.assign_coords(time_bnds=(['time', 'bnds'], [[t1, t2]]))
        time_bnds_var = dataset.coords['time_bnds']
        time_bnds_var.attrs['long_name'] = 'time'
        time_bnds_var.attrs['standard_name'] = 'time'
        time_bnds_var.attrs['units'] = DATETIME_UNITS
        time_bnds_var.attrs['calendar'] = DATETIME_CALENDAR
        time_bnds_var.encoding['units'] = DATETIME_UNITS
        time_bnds_var.encoding['calendar'] = DATETIME_CALENDAR
    return dataset


def get_time_in_days_since_1970(time_str: str, pattern=None) -> float:
    datetime = pd.to_datetime(time_str, format=pattern, infer_datetime_format=True, utc=True)
    timedelta = datetime - REF_DATETIME
    return timedelta.days + timedelta.seconds / SECONDS_PER_DAY + timedelta.microseconds / MICROSECONDS_PER_DAY


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


def sort_by_time(input_paths: Sequence[str], input_reader, input_processor, monitor):
    sorted_input_paths = []
    times = []
    input_paths = [input_file for f in input_paths for input_file in glob.glob(f, recursive=True)]
    for input_path in input_paths:
        try:
            input_dataset = input_reader.read(input_path)
            monitor(f'Dataset read for sorting by time:\n{input_path}')
        except Exception as e:
            monitor(f'ERROR: cannot read input: {e}: skipping...')
            traceback.print_exc()
            return False
        time_range = input_processor.get_time_range(input_dataset)
        if _get_half_time(time_range) not in times:
            sorted_input_paths.append(input_path)
            times.append(_get_half_time(time_range))

    times, sorted_input_paths = (list(t) for t in zip(*sorted(zip(times, sorted_input_paths))))

    return sorted_input_paths


def _get_half_time(time_range):
    start_time = time_range[0]
    end_time = time_range[0]
    half_seconds = (end_time - start_time) / 2
    halftime = start_time + half_seconds
    return halftime