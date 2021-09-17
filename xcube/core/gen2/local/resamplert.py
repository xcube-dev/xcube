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

import pandas as pd
import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.core.resampling import resample_in_time
from xcube.util.assertions import assert_instance
from .transformer import CubeTransformer
from .transformer import TransformedCube
from ..config import CubeConfig


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

        if cube_config.time_period is None:
            resampled_cube = cube
        else:
            time_resample_params = dict()
            time_resample_params['frequency'] = cube_config.time_period
            time_resample_params['method'] = 'first'
            if self._time_range:
                start_time = pd.to_datetime(self._time_range[0])
                dataset_start_time = cube.time[0].values
                time_delta = dataset_start_time - start_time
                time_resample_params['offset'] = time_delta
            if cube_config.temporal_resampling is not None:
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
        cube_config = cube_config.drop_props(['time_period'])

        return resampled_cube, gm, cube_config
