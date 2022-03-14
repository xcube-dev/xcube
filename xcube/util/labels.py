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

    If var has a 'time' dimension of type datetime64 and labels has a 'time'
    key with a timezone-aware value, return a modified labels dictionary with
    a timezone-naive time value. Otherwise return the original labels.
    """
    if _has_datetime64_time(var) and \
       'time' in labels and pd.Timestamp(labels['time']).tzinfo is not None:
        naive_time = pd.Timestamp(labels['time']).tz_convert(None)
        return dict(labels, time=naive_time)
    else:
        return labels


def _has_datetime64_time(var: xr.DataArray) -> bool:
    """Report whether var has a time dimension with type datetime64"""
    return 'time' in var.dims and \
           hasattr(var['time'], 'dtype') and \
           hasattr(var['time'].dtype, 'type') and \
           var['time'].dtype.type is np.datetime64