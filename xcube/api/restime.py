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

from typing import Dict, Any, Sequence

import xarray as xr

from xcube.util.dsutil import select_variables

RESAMPLING_METHODS = ['all', 'any', 'argmin', 'argmax', 'count', 'first', 'last', 'max', 'mean', 'median', 'min',
                      'backfill', 'bfill', 'ffill', 'interpolate', 'nearest', 'pad']


def resample_in_time(cube: xr.Dataset,
                     frequency: str,
                     resampling: str,
                     var_names: Sequence[str] = None,
                     metadata: Dict[str, Any] = None):
    """
    Resample a data cube in the time dimension.

    :param cube: The data cube.
    :param frequency: Resampling frequency.
    :param resampling: Resampling method.
    :param var_names: Variable names to include.
    :param metadata: Output metadata.
    :return: A new data cube resampled in time.
    """
    if var_names:
        cube = select_variables(cube, var_names)

    resampler = cube.resample(skipna=True,
                              closed='left',
                              label='left',
                              keep_attrs=True,
                              time=frequency)

    resampling_method = getattr(resampler, resampling)
    resampled_ds = resampling_method('time')

    # TODO: add time_bnds to resampled_ds
    time_coverage_start = '%s' % cube.time[0]
    time_coverage_end = '%s' % cube.time[-1]

    resampled_ds.attrs.update(metadata or {})
    # TODO: add other time_coverage_ attributes
    resampled_ds.attrs.update(time_coverage_start=time_coverage_start,
                              time_coverage_end=time_coverage_end)
    return resampled_ds
