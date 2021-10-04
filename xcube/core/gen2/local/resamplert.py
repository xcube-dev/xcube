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

import pandas as pd
import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.core.resampling import resample_in_time
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
                        ['linear', 'nearest', 'nearest-up', 'zero', 'slinear',
                         'quadratic', 'cubic', 'previous', 'next']:
                    time_resample_params['method'] = 'interp'
                    time_resample_params['interp_kind'] = \
                        cube_config.temporal_resampling
                else:
                    time_resample_params['method'] = \
                        cube_config.temporal_resampling
            resampled_cube = resample_in_time(
                cube,
                rename_variables=False,
                **time_resample_params
            )

        cube_config = cube_config.drop_props(to_drop)

        return resampled_cube, gm, cube_config


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