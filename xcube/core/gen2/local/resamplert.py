# Copyright (c) 2021 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
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
import pandas as pd
import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.core.resampling import resample_in_time
from xcube.core.resampling.temporal import adjust_metadata_and_chunking
from xcube.util.assertions import assert_instance
from .transformer import CubeTransformer
from .transformer import TransformedCube
from ..config import CubeConfig
from ..error import CubeGeneratorError


class CubeResamplerT(CubeTransformer):

    def __init__(self,
                 cube_config: CubeConfig):
        assert_instance(cube_config, CubeConfig, 'cube_config')
        self._time_range = cube_config.time_range \
            if cube_config.time_range else None

    def transform_cube(self,
                       cube: xr.Dataset,
                       gm: GridMapping,
                       cube_config: CubeConfig) -> TransformedCube:
        to_drop = []
        if cube_config.time_period is None:
            resampled_cube = cube
        else:
            to_drop.append('time_period')
            time_resample_params = dict()
            time_resample_params['frequency'] = cube_config.time_period
            time_resample_params['method'] = 'first'
            if self._time_range:
                import re
                time_unit = re.findall('[A-Z]+', cube_config.time_period)[0]
                if time_unit in ['H', 'D']:
                    start_time = pd.to_datetime(self._time_range[0])
                    dataset_start_time = pd.Timestamp(cube.time[0].values)
                    time_delta = _normalize_time(dataset_start_time) \
                        - start_time
                    period_delta = pd.Timedelta(cube_config.time_period)
                    if time_delta > period_delta:
                        if time_unit == 'H':
                            time_resample_params['base'] = \
                                time_delta.hours / period_delta.hours
                        elif time_unit == 'D':
                            time_resample_params['base'] = \
                                time_delta.days / period_delta.days
            if cube_config.temporal_resampling is not None:
                to_drop.append('temporal_resampling')
                if cube_config.temporal_resampling in \
                        ['linear', 'nearest-up', 'zero', 'slinear',
                         'quadratic', 'cubic', 'previous', 'next']:
                    time_resample_params['method'] = 'interpolate'
                    time_resample_params['interp_kind'] = \
                        cube_config.temporal_resampling
                else:
                    time_resample_params['method'] = \
                        cube_config.temporal_resampling
            # we set cub_asserted to true so the resampling can deal with
            # cftime data
            resampled_cube = resample_in_time(
                cube,
                rename_variables=False,
                cube_asserted=True,
                **time_resample_params
            )
        if self._time_range:
            # cut possible overlapping time steps
            is_cf_time = isinstance(resampled_cube.time_bnds[0].values[0],
                                    cftime.datetime)
            if is_cf_time:
                resampled_cube = _get_temporal_subset_cf(resampled_cube,
                                                         self._time_range)
            else:
                resampled_cube = _get_temporal_subset(resampled_cube,
                                                      self._time_range)
            adjust_metadata_and_chunking(resampled_cube, time_chunk_size=1)

        cube_config = cube_config.drop_props(to_drop)

        return resampled_cube, gm, cube_config


def _get_temporal_subset_cf(resampled_cube, time_range):
    try:
        data_start_index = resampled_cube.time_bnds[:, 0].to_index().\
            get_loc(time_range[0], method='bfill')
        if isinstance(data_start_index, slice):
            data_start_index = data_start_index.start
    except KeyError:
        data_start_index = 0
    try:
        data_end_index = resampled_cube.time_bnds[:, 1].to_index().\
            get_loc(time_range[1], method='ffill')
        if isinstance(data_end_index, slice):
            data_end_index = data_end_index.stop + 1
    except KeyError:
        data_end_index = resampled_cube.time.size
    return resampled_cube.isel(time=slice(data_start_index, data_end_index))


def _get_temporal_subset(resampled_cube, time_range):
    try:
        data_start_time = resampled_cube.time_bnds[:, 0]. \
            sel(time=time_range[0], method='bfill')
        if data_start_time.size < 1:
            data_start_time = resampled_cube.time_bnds[0, 0]
    except KeyError:
        data_start_time = resampled_cube.time_bnds[0, 0]
    try:
        data_end_time = resampled_cube.time_bnds[:, 1]. \
            sel(time=time_range[1], method='ffill')
        if data_end_time.size < 1:
            data_end_time = resampled_cube.time_bnds[-1, 1]
    except KeyError:
        data_end_time = resampled_cube.time_bnds[-1, 1]
    return resampled_cube.sel(time=slice(data_start_time, data_end_time))


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
