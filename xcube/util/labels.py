# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Dict, Any

import numpy as np
import pandas as pd
import xarray as xr


def ensure_time_compatible(var: xr.DataArray,
                           labels: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure that labels['time'] is timezone-naive, if necessary.

    This function returns either the passed-in labels object, or a copy of
    it with a modified value for labels['time'].

    If there is no 'time' key in the labels dictionary or if there is no
    'time' dimension in the var array, the original labels are returned.

    If there is a 'time' key, it is expected that its value will be
    a valid timestamp (i.e. a valid input to pandas.Timestamp.__init__), or
    a slice in which the start and stop fields are valid timestamps. For a
    slice, the start and stop fields are processed separately, and their
    modified values (if required) are returned as the start and stop fields
    of a new slice. The step field is included unchanged in the new slice.

    If var has a 'time' dimension of type datetime64 and labels has a 'time'
    key with a timezone-aware value, return a modified labels dictionary with
    a timezone-naive time value. Otherwise return the original labels.

    """

    if 'time' not in labels or 'time' not in var.dims:
        return labels

    timeval = labels['time']
    if isinstance(timeval, slice):
        # process start and stop separately and pass step through unchanged
        return dict(labels, time=slice(
            _ensure_timestamp_compatible(var, timeval.start),
            _ensure_timestamp_compatible(var, timeval.stop),
            timeval.step))
    else:
        return dict(labels, time=_ensure_timestamp_compatible(var, timeval))


def _ensure_timestamp_compatible(var: xr.DataArray, timeval: Any):
    timestamp = pd.Timestamp(timeval)
    timeval_is_naive = timestamp.tzinfo is None
    if _has_datetime64_time(var) and not timeval_is_naive:
        # pandas treats datetime64s as timezone-naive, so we naivefy the label
        naive_time = timestamp.tz_convert(None)
        return naive_time
    else:
        return timeval


def _has_datetime64_time(var: xr.DataArray) -> bool:
    """Report whether var has a time dimension with type datetime64

    Assumes a 'time' key is present in var.dims."""
    return hasattr(var['time'], 'dtype') \
        and hasattr(var['time'].dtype, 'type') \
        and var['time'].dtype.type is np.datetime64
