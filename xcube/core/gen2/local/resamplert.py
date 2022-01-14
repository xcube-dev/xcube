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

import cftime
import numpy as np
import pandas as pd
import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.core.resampling import resample_in_time
from xcube.core.resampling.temporal import adjust_metadata_and_chunking
from xcube.core.resampling.temporal import INTERPOLATION_KINDS
from xcube.core.timecoord import get_time_range_from_data
from .transformer import CubeTransformer
from .transformer import TransformedCube
from ..config import CubeConfig
from ..error import CubeGeneratorError

MIN_MAX_DELTAS = dict(
    H=(1, 1, 'H'),
    D=(1, 1, 'D'),
    W=(7, 7, 'D'),
    M=(28, 31, 'D'),
    Y=(365, 366, 'D')
)


class CubeResamplerT(CubeTransformer):

    def transform_cube(self,
                       cube: xr.Dataset,
                       gm: GridMapping,
                       cube_config: CubeConfig) -> TransformedCube:
        to_drop = []
        if cube_config.time_range is not None:
            start_time, end_time = cube_config.time_range
            to_drop.append('time_range')
        else:
            start_time, end_time = \
                get_time_range_from_data(cube, maybe_consider_metadata=False)
        if cube_config.time_period is None:
            resampled_cube = cube
        else:
            to_drop.append('time_period')
            time_resample_params = dict()
            time_resample_params['frequency'] = cube_config.time_period
            time_resample_params['method'] = 'first'
            import re
            time_unit = re.findall('[A-Z]+', cube_config.time_period)[0]
            time_frequency = int(cube_config.time_period.split(time_unit)[0])
            if time_unit in ['H', 'D']:
                if start_time is not None:
                    start_time_as_datetime = pd.to_datetime(start_time)
                    dataset_start_time = pd.Timestamp(cube.time[0].values)
                    time_delta = _normalize_time(dataset_start_time) \
                        - start_time_as_datetime
                    _adjust_time_resample_params(time_resample_params,
                                                 cube_config.time_period,
                                                 time_delta,
                                                 time_unit)
                elif end_time is not None:
                    end_time_as_datetime = pd.to_datetime(end_time)
                    dataset_end_time = pd.Timestamp(cube.time[-1].values)
                    time_delta = end_time_as_datetime - \
                        _normalize_time(dataset_end_time)
                    _adjust_time_resample_params(time_resample_params,
                                                 cube_config.time_period,
                                                 time_delta,
                                                 time_unit)
            if cube_config.temporal_resampling is not None:
                to_drop.append('temporal_resampling')
                min_data_delta, max_data_delta = \
                    get_min_max_timedeltas_from_data(cube)
                min_period_delta, max_period_delta = \
                    get_min_max_timedeltas_for_time_period(time_frequency,
                                                           time_unit)
                if max_data_delta < min_period_delta:
                    if 'downsampling' not in cube_config.temporal_resampling:
                        raise ValueError('Data must be sampled down to a'
                                         'coarser temporal resolution, '
                                         'but no temporal downsampling '
                                         'method is set')
                    try:
                        method, method_args = \
                            cube_config.temporal_resampling['downsampling']
                    except ValueError:
                        method = cube_config.temporal_resampling['downsampling']
                        method_args = {}
                elif max_period_delta < min_data_delta:
                    if 'upsampling' not in cube_config.temporal_resampling:
                        raise ValueError('Data must be sampled up to a'
                                         'finer temporal resolution, '
                                         'but no temporal upsampling '
                                         'method is set')
                    try:
                        method, method_args = \
                            cube_config.temporal_resampling['upsampling']
                    except ValueError:
                        method = cube_config.temporal_resampling['upsampling']
                        method_args = {}
                else:
                    if 'downsampling' not in cube_config.temporal_resampling \
                            and 'upsampling' not in \
                            cube_config.temporal_resampling:
                        raise ValueError('Please specify a method for temporal '
                                         'resampling.')
                    if 'downsampling' in cube_config.temporal_resampling and \
                            'upsampling' in cube_config.temporal_resampling:
                        raise ValueError('Cannot determine unambiguously '
                                         'whether data needs to be sampled up '
                                         'or down temporally. Please only '
                                         'specify one method for temporal '
                                         'resampling.')
                    try:
                        method, method_args = cube_config.temporal_resampling.\
                            get('downsampling',
                                cube_config.temporal_resampling.
                                get('upsampling'))
                    except ValueError:
                        method = cube_config.temporal_resampling.get(
                            'downsampling',
                            cube_config.temporal_resampling.get('upsampling'))
                        method_args = {}
                if method == 'interpolate':
                    time_resample_params['method'] = method
                    if 'kind' not in method_args:
                        interpolation_kinds = \
                            ', '.join(map(repr, INTERPOLATION_KINDS))
                        raise ValueError(f"To use 'interpolation' as "
                                         f"upsampling method, the "
                                         f"interpolation kind must be set. "
                                         f"Use any of the following: "
                                         f"{interpolation_kinds}.")
                    if method_args['kind'] not in INTERPOLATION_KINDS:
                        interpolation_kinds = \
                            ', '.join(map(repr, INTERPOLATION_KINDS))
                        raise ValueError(f'Interpolation kind must be one of '
                                         f'the following: '
                                         f'{interpolation_kinds}. Was: '
                                         f'"{method_args["kind"]}".')
                    time_resample_params['interp_kind'] = method_args['kind']
                elif method == 'percentile':
                    if 'threshold' not in method_args:
                        raise ValueError(f"To use 'percentile' as "
                                         f"downsampling method, a "
                                         f"threshold must be set.")
                    method = f'percentile_{method_args["threshold"]}'
                    time_resample_params['method'] = method
                else:
                    time_resample_params['method'] = method
            # we set cube_asserted to true so the resampling can deal with
            # cftime data
            resampled_cube = resample_in_time(
                cube,
                rename_variables=False,
                cube_asserted=True,
                **time_resample_params
            )
        if start_time is not None or end_time is not None:
            # cut possible overlapping time steps
            is_cf_time = isinstance(resampled_cube.time_bnds[0].values[0],
                                    cftime.datetime)
            if is_cf_time:
                resampled_cube = _get_temporal_subset_cf(resampled_cube,
                                                         start_time,
                                                         end_time)
            else:
                resampled_cube = _get_temporal_subset(resampled_cube,
                                                      start_time,
                                                      end_time)
            adjust_metadata_and_chunking(resampled_cube, time_chunk_size=1)

        cube_config = cube_config.drop_props(to_drop)

        return resampled_cube, gm, cube_config


def _adjust_time_resample_params(time_resample_params,
                                 time_period,
                                 time_delta,
                                 time_unit):
    period_delta = pd.Timedelta(time_period)
    if time_delta > period_delta:
        if time_unit == 'H':
            time_resample_params['base'] = \
                time_delta.hours / period_delta.hours
        elif time_unit == 'D':
            time_resample_params['base'] = \
                time_delta.days / period_delta.days


def _get_temporal_subset_cf(resampled_cube, start_time, end_time):
    data_start_index = 0
    data_end_index = resampled_cube.time.size
    if start_time:
        try:
            data_start_index = resampled_cube.time_bnds[:, 0].to_index().\
                get_loc(start_time, method='bfill')
            if isinstance(data_start_index, slice):
                data_start_index = data_start_index.start
        except KeyError:
            pass
    if end_time:
        try:
            data_end_index = resampled_cube.time_bnds[:, 1].to_index().\
                get_loc(end_time, method='ffill')
            if isinstance(data_end_index, slice):
                data_end_index = data_end_index.stop + 1
        except KeyError:
            pass
    return resampled_cube.isel(time=slice(data_start_index, data_end_index))


def _get_temporal_subset(resampled_cube, start_time, end_time):
    data_start_time = resampled_cube.time_bnds[0, 0]
    data_end_time = resampled_cube.time_bnds[-1, 1]
    if start_time:
        try:
            data_start_time = resampled_cube.time_bnds[:, 0]. \
                sel(time=start_time, method='bfill')
            if data_start_time.size < 1:
                data_start_time = resampled_cube.time_bnds[0, 0]
        except KeyError:
            pass
    if end_time:
        try:
            data_end_time = resampled_cube.time_bnds[:, 1]. \
                sel(time=end_time, method='ffill')
            if data_end_time.size < 1:
                data_end_time = resampled_cube.time_bnds[-1, 1]
        except KeyError:
            pass
    return resampled_cube.sel(time=slice(data_start_time, data_end_time))


def get_min_max_timedeltas_from_data(data: xr.Dataset):
    time_diff = data['time'].diff(dim=data['time'].dims[0])\
        .values.astype(np.float64)
    return pd.Timedelta(min(time_diff)), pd.Timedelta(max(time_diff))


def get_min_max_timedeltas_for_time_period(time_frequency: int, time_unit: str):
    min_freq = MIN_MAX_DELTAS[time_unit][0] * time_frequency
    max_freq = MIN_MAX_DELTAS[time_unit][1] * time_frequency
    delta_unit = MIN_MAX_DELTAS[time_unit][2]
    return pd.Timedelta(f'{min_freq}{delta_unit}'), \
        pd.Timedelta(f'{max_freq}{delta_unit}')


def _normalize_time(time, normalize_hour=True):
    if normalize_hour:
        return time.replace(hour=0, minute=0, second=0, microsecond=0,
                            nanosecond=0)
    return time.replace(minute=0, second=0, microsecond=0, nanosecond=0)


def _get_expected_start_time(dataset_start_time, time_unit):
    if time_unit == 'H':
        return _normalize_time(dataset_start_time, normalize_hour=False)
    if time_unit == 'D':
        return _normalize_time(dataset_start_time)
    if time_unit == 'W':
        delta = pd.Timedelta(-dataset_start_time.day_of_week)
        return _normalize_time(dataset_start_time) - delta
    if time_unit == 'M':
        return _normalize_time(dataset_start_time).replace(day=1)
    if time_unit == 'Q':
        delta = pd.Timedelta(-(dataset_start_time.month - 1) % 3)
        return _normalize_time(dataset_start_time).replace(day=1) - delta
    if time_unit == 'Y':
        return _normalize_time(dataset_start_time).replace(month=1, day=1)
    raise CubeGeneratorError(f'Unsupported time unit "{time_unit}"')
