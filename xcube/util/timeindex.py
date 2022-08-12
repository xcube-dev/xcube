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

"""Utilities to ensure compatibility between time variables and their indexers

The utilities in this module check and, where necessary, modify time indexers
to ensure that they are compatible with the variables they are indexing.
"""

from typing import Dict, Any, Union

import numpy as np
import pandas as pd
import xarray as xr
import logging

logger = logging.getLogger('xcube')


def ensure_time_label_compatible(var: Union[xr.DataArray, xr.Dataset],
                                 labels: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure that labels['time'] is compatible with var

    This function returns either the passed-in labels object, or a copy of
    it with a modified value for labels['time'].

    If there is no 'time' key in the labels dictionary or if there is no
    'time' dimension in the var array, the original labels are returned.

    If there is a 'time' key in the labels dictionary and a 'time' dimension
    in the var array, they are checked for compatibility. If they are
    compatible, the original labels are returned. If not, an altered labels
    dictionary is returned, in which the time key has been modified to be
    compatible with the type of the 'time' dimension in the var array.

    See the documentation for `ensure_time_index_compatible` for details on
    the compatibility check.
    """

    if 'time' in labels and 'time' in var.dims:
        return dict(labels,
                    time=ensure_time_index_compatible(var, labels['time']))
    else:
        return labels


def ensure_time_index_compatible(var: Union[xr.DataArray, xr.Dataset],
                                 timeval: Any) -> Any:
    """Ensure that timeval is a valid time indexer for var

    It is expected that the value of timeval will be a valid timestamp (i.e.
    a valid input to pandas.Timestamp.__init__), or a slice in which the
    start and stop fields are valid timestamps. For a slice, the start and
    stop fields are processed separately, and their modified values (if
    required) are returned as the start and stop fields of a new slice. The
    step field is included unchanged in the new slice.

    The compatibility check consists of checking the timezone-awareness
    of the variable and its indexer. If the variable is timezone-aware and
    the indexer is timezone-naive, or vice versa, a new indexer is returned
    whose timezone-awareness corresponds to that of the variable. Otherwise
    the original indexer is returned.
    """
    
    if isinstance(timeval, slice):
        # process start and stop separately, and pass step through unchanged
        return slice(
            _ensure_timestamp_compatible(var, timeval.start),
            _ensure_timestamp_compatible(var, timeval.stop),
            timeval.step)
    else:
        return _ensure_timestamp_compatible(var, timeval)


def _ensure_timestamp_compatible(var: xr.DataArray, timeval: Any):
    if timeval is None:
        return None

    if hasattr(timeval, 'tzinfo'):
        timestamp = timeval
        timeval_tzinfo = timeval.tzinfo
    else:
        try:
            timestamp = pd.Timestamp(timeval)
            timeval_tzinfo = timestamp.tzinfo
        except (TypeError, ValueError):
            logger.warning('Can\'t determine indexer timezone, leaving it '
                           'unmodified.')
            return timeval

    if _has_datetime64_time(var):
        # pandas treats all datetime64 arrays as timezone-naive
        array_timezone = None
    elif hasattr(var.time[0:1].values[0], 'tzinfo'):
        array_timezone = var.time[0:1].values[0].tzinfo
    else:
        logger.warning(
            'Can\'t determine array timezone, leaving indexer unmodified.'
        )
        return timeval

    if array_timezone is None and timeval_tzinfo is not None:
        if hasattr(timestamp, 'tz_convert'):
            return timestamp.tz_convert(None)
        else:
            logger.warning('Indexer lacks tz_convert, leaving unmodified')
            return timeval
    elif array_timezone is not None and timeval_tzinfo is None:
        if hasattr(timestamp, 'tz_localize'):
            return timestamp.tz_localize(array_timezone)
        else:
            logger.warning('Indexer lacks tz_localize, leaving unmodified')
            return timeval
    else:
        return timeval


def _has_datetime64_time(var: xr.DataArray) -> bool:
    """Report whether var's time dimension has type datetime64

    Assumes that a 'time' key is present in var.dims."""
    return hasattr(var['time'], 'dtype') \
           and hasattr(var['time'].dtype, 'type') \
           and var['time'].dtype.type is np.datetime64