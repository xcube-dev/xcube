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

from enum import Enum
from typing import Dict, Any, Sequence, Union

import cftime
from datetime import timedelta
import numpy as np
import xarray as xr
from xarray.coding.cftime_offsets import Day
from xarray.coding.cftime_offsets import Hour
from xarray.coding.cftime_offsets import MonthBegin
from xarray.coding.cftime_offsets import QuarterBegin
from xarray.coding.cftime_offsets import YearBegin

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


class Offset(Enum):
    PREVIOUS = 'previous'
    NONE = 'none'
    NEXT = 'next'


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

    frequency_is_irregular = frequency.endswith('Y') or \
                         frequency.endswith('M') or \
                         frequency.endswith('Q')
    # resample to start of period
    if frequency_is_irregular:
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
    if method in UPSAMPLING_METHODS:
        resampled_cube = _adjust_upsampled_cube(resampled_cube,
                                                frequency,
                                                base,
                                                frequency_is_irregular)
    else:
        resampled_cube = _adjust_downsampled_cube(resampled_cube,
                                                  frequency,
                                                  base,
                                                  frequency_is_irregular)
    return adjust_metadata_and_chunking(resampled_cube,
                                        metadata=metadata,
                                        time_chunk_size=time_chunk_size)


def _adjust_upsampled_cube(resampled_cube, frequency, base, frequency_is_irregular):
    # Times of upsampled cube are correct, we need to determine time bounds
    # Get times with negative offset
    times = resampled_cube.time.values
    previous_times = _get_resampled_times(
        resampled_cube, frequency, 'time', Offset.PREVIOUS, base
    )
    # Get centers between times and previous_times as start bounds
    center_times = _get_centers_between_times(
        previous_times,
        times,
        frequency_is_irregular,
        resampled_cube
    )
    # we need to add this as intermediate data array so we can retrieve
    # resampled times from it
    resampled_cube = resampled_cube.assign_coords(
        intermediate_time=center_times
    )
    stop_times = _get_resampled_times(
        resampled_cube, frequency, 'intermediate_time', Offset.NEXT, base
    )
    resampled_cube = resampled_cube.drop_vars('intermediate_time')
    resampled_cube = _add_time_bounds_to_resampled_cube(center_times,
                                                        stop_times,
                                                        resampled_cube)
    return resampled_cube


def _adjust_downsampled_cube(resampled_cube,
                             frequency,
                             base,
                             frequency_is_irregular):
    # times of resampled_cube are actually start bounding times.
    # We need to determine times and end bounding times
    start_times = resampled_cube.time.values
    stop_times = _get_resampled_times(
        resampled_cube, frequency, 'time', Offset.NEXT, base
    )
    resampled_cube = _add_time_bounds_to_resampled_cube(start_times,
                                                        stop_times,
                                                        resampled_cube)
    # Get centers between start and stop bounding times
    center_times = _get_centers_between_times(
        start_times,
        stop_times,
        frequency_is_irregular,
        resampled_cube
    )
    resampled_cube = resampled_cube.assign_coords(time=center_times)
    return resampled_cube


def _get_resampled_times(cube: xr.Dataset,
                         frequency: str,
                         name_of_time_dim: str,
                         offset: Offset,
                         base=None):
    if offset == Offset.PREVIOUS:
        offset = _invert_frequency(frequency,
                                   cube[name_of_time_dim].values[0])
    elif offset == Offset.NONE:
        offset = None
    elif offset == Offset.NEXT:
        offset = frequency
    args = dict(skipna=True,
                closed='left',
                label='left',
                loffset=offset,
                base=base)
    args[name_of_time_dim] = frequency
    return np.array(list(cube[name_of_time_dim].resample(**args).groups.keys()))


def _add_time_bounds_to_resampled_cube(start_times, stop_times, resampled_cube):
    time_bounds = xr.DataArray(
        np.array([start_times, stop_times]).transpose(),
        dims=['time', 'bnds']
    )
    return resampled_cube.assign_coords(
        time_bnds=time_bounds
    )


def _get_centers_between_times(earlier_times,
                               later_times,
                               frequency_is_irregular,
                               resampled_cube):
    """
    Determines the center between two time arrays.
    In case the frequency is irregular and the centers are close to the
    beginning of a month, the centers are snapped to it
    """
    time_deltas = later_times - earlier_times
    center_times = later_times - time_deltas * 0.5
    if frequency_is_irregular:
        # In case of 'M', 'Q' or 'Y' frequencies, add a small time delta
        # so we move a little closer to the later time
        time_delta = _get_time_delta(earlier_times[0])
        center_times_plus_delta = center_times + time_delta
        resampled_cube = resampled_cube.assign_coords(
            intermediate_time=center_times_plus_delta
        )
        # snap center times to beginnings of months when they are close
        starts_of_month = _get_resampled_times(
            resampled_cube, '1MS', 'intermediate_time', Offset.NONE
        )
        center_time_deltas = center_times_plus_delta - starts_of_month
        snapped_times = np.where(center_time_deltas < time_delta * 2,
                                 starts_of_month,
                                 center_times)
        resampled_cube.drop_vars('intermediate_time')
        return snapped_times
    return center_times


def _get_time_delta(time_value):
    if _is_cf(time_value):
        return timedelta(hours=42)
    return np.timedelta64(42, 'h')


def _invert_frequency(frequency, time_value):
    if not _is_cf(time_value):
        return f'-{frequency}'
    if frequency.endswith('H'):
        frequency_value = frequency.split('H')[0]
        return Hour(-int(frequency_value))
    if frequency.endswith('D'):
        frequency_value = frequency.split('D')[0]
        return Day(-int(frequency_value))
    if frequency.endswith('W'):
        frequency_value = frequency.split('W')[0]
        return Day(-int(frequency_value) * 7)
    if frequency.endswith('MS'):
        frequency_value = frequency.split('MS')[0]
        return MonthBegin(-int(frequency_value))
    if frequency.endswith('QS'):
        frequency_value = frequency.split('QS')[0]
        return QuarterBegin(-int(frequency_value))
    frequency_value = frequency.split('YS')[0]
    return YearBegin(-int(frequency_value))


def _is_cf(time_value):
    return isinstance(time_value, cftime.datetime)


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
