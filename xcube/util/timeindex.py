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
                                 labels: Dict[str, Any],
                                 time_name: str = 'time') -> Dict[str, Any]:
    """Ensure that *labels[time_name]* is compatible with *var*

    This function returns either the passed-in *labels* object, or a copy of
    it with a modified value for *labels[time_name]*.

    The parameter *time_name* specifies the name of the variable representing
    time, and defaults to ``'time'``.

    If there is no *time_name* key in the labels dictionary or if there is no
    *time_name* dimension in the var array, the original labels are returned.

    If there is a *time_name* key in the labels dictionary and a *time_name*
    dimension in the var array, they are checked for compatibility. If they
    are compatible, the original labels are returned. If not, an altered
    labels dictionary is returned, in which the time key has been modified to
    be compatible with the type of the *time_name* dimension in the var array.

    See the documentation for *ensure_time_index_compatible* for details on
    the compatibility check.
    """

    if time_name in labels and time_name in var.dims:
        return dict(labels,
                    time=ensure_time_index_compatible(var, labels[time_name],
                                                      time_name))
    else:
        return labels


def ensure_time_index_compatible(var: Union[xr.DataArray, xr.Dataset],
                                 time_value: Any,
                                 time_name: str = 'time') -> Any:
    """Ensure that *time_value* is a valid time indexer for *var*

    It is expected that the value of *time_value* will be a valid timestamp
    (i.e. a valid input to pandas.Timestamp.__init__), or a slice in which
    the start and stop fields are valid timestamps. For a slice, the start
    and stop fields are processed separately, and their modified values (if
    required) are returned as the start and stop fields of a new slice. The
    step field is included unchanged in the new slice.

    The compatibility check consists of checking the timezone-awareness
    of the variable and its indexer. If the variable is timezone-aware and
    the indexer is timezone-naive, or vice versa, a new indexer is returned
    whose timezone-awareness corresponds to that of the variable. Otherwise
    the original indexer is returned.

    Purpose: xarray throws an error if one attempts to index a timezone-aware
    variable with a timezone-naive indexer, or vice versa. This function can
    be called before indexing to align the indexer's timezone-awareness with
    that of the variable, thus avoiding the error from xarray.
    """
    
    if isinstance(time_value, slice):
        # process start and stop separately, and pass step through unchanged
        return slice(
            _ensure_timestamp_compatible(var, time_value.start, time_name),
            _ensure_timestamp_compatible(var, time_value.stop, time_name),
            time_value.step)
    else:
        return _ensure_timestamp_compatible(var, time_value, time_name)


def _ensure_timestamp_compatible(var: xr.DataArray, time_value: Any,
                                 time_name: str):
    if time_value is None:
        return None

    if hasattr(time_value, 'tzinfo'):
        timestamp = time_value
        time_value_tzinfo = time_value.tzinfo
    else:
        try:
            timestamp = pd.Timestamp(time_value)
            time_value_tzinfo = timestamp.tzinfo
        except (TypeError, ValueError):
            logger.warning('Can\'t determine indexer timezone, leaving it '
                           'unmodified.')
            return time_value

    if _has_datetime64_time(var, time_name):
        # pandas treats all datetime64 arrays as timezone-naive
        array_timezone = None
    elif hasattr(var.time[0:1].values[0], 'tzinfo'):
        array_timezone = var.time[0:1].values[0].tzinfo
    else:
        logger.warning(
            'Can\'t determine array timezone, leaving indexer unmodified.'
        )
        return time_value

    if array_timezone is None and time_value_tzinfo is not None:
        if hasattr(timestamp, 'tz_convert'):
            return timestamp.tz_convert(None)
        else:
            logger.warning('Indexer lacks tz_convert, leaving unmodified')
            return time_value
    elif array_timezone is not None and time_value_tzinfo is None:
        if hasattr(timestamp, 'tz_localize'):
            return timestamp.tz_localize(array_timezone)
        else:
            logger.warning('Indexer lacks tz_localize, leaving unmodified')
            return time_value
    else:
        return time_value


def _has_datetime64_time(var: xr.DataArray, time_name) -> bool:
    """Report whether *var*'s time dimension has type ``datetime64``

    *time_name* specifies the name of the time dimension.

    It is assumed that a *time_name* key is present in var.dims."""
    return hasattr(var[time_name], 'dtype') \
           and hasattr(var[time_name].dtype, 'type') \
           and var[time_name].dtype.type is np.datetime64
