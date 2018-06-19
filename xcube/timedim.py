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
from typing import Tuple

import pandas as pd
import xarray as xr

REF_DATETIME_STR = '1970-01-01 00:00:00'
REF_DATETIME = pd.to_datetime(REF_DATETIME_STR)
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
    datetime = pd.to_datetime(time_str, format=pattern, infer_datetime_format=True)
    timedelta = datetime - REF_DATETIME
    return timedelta.days + timedelta.seconds / SECONDS_PER_DAY + timedelta.microseconds / MICROSECONDS_PER_DAY


def _assert(cond, text='Assertion failed'):
    if not cond:
        raise ValueError(text)
