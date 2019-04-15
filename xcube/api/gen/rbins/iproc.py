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

import datetime
from typing import Tuple, Optional

import numpy as np
import xarray as xr

from xcube.util.constants import CRS_WKT_EPSG_4326
from xcube.util.timecoord import get_time_in_days_since_1970
from ..iproc import XYInputProcessor, ReprojectionInfo, register_input_processor


class RbinsSeviriHighrocSceneInputProcessor(XYInputProcessor):
    """
    Input processor for RBINS' HIGHROC single-scene Level-2 NetCDF inputs.
    """

    @property
    def name(self) -> str:
        return 'rbins-seviri-highroc-scene-l2'

    @property
    def description(self) -> str:
        return 'RBINS SEVIRI HIGHROC single-scene Level-2 NetCDF inputs'

    @property
    def input_reader(self) -> str:
        return 'netcdf4'

    def get_reprojection_info(self, dataset: xr.Dataset) -> ReprojectionInfo:
        return ReprojectionInfo(xy_var_names=('lon', 'lat'),
                                xy_crs=CRS_WKT_EPSG_4326,
                                xy_gcp_step=1)

    def get_time_range(self, dataset: xr.Dataset) -> Tuple[float, float]:
        date = dataset.attrs.get('DATE')
        if date is None:
            raise ValueError('illegal L2 input: missing DATE attribute')
        time = dataset.attrs.get('TIME', '1200')
        days_since_1970 = get_time_in_days_since_1970(date + time)
        return days_since_1970, days_since_1970


class RbinsSeviriHighrocDailyInputProcessor(XYInputProcessor):
    """
    Input processor for RBINS' HIGHROC daily Level-2 NetCDF inputs.
    """

    @property
    def name(self) -> str:
        return 'rbins-seviri-highroc-daily-l2'

    @property
    def description(self) -> str:
        return 'RBINS SEVIRI HIGHROC daily Level-2 NetCDF inputs'

    @property
    def input_reader(self) -> str:
        return 'netcdf4'

    def get_reprojection_info(self, dataset: xr.Dataset) -> ReprojectionInfo:
        return ReprojectionInfo(xy_var_names=('longitude', 'latitude'),
                                xy_crs=CRS_WKT_EPSG_4326,
                                xy_gcp_step=1)

    def get_time_range(self, dataset: xr.Dataset) -> Optional[Tuple[float, float]]:
        return None

    def pre_process(self, dataset: xr.Dataset) -> xr.Dataset:
        num_times = dataset.sizes.get('t')
        time = np.ndarray(shape=num_times, dtype=np.dtype('datetime64[us]'))
        for i in range(num_times):
            date = dataset.DATE[i]
            hour = dataset.HOUR[i]
            minute = dataset.MIN[i]
            year = date // 10000
            month = (date - year * 10000) // 100
            day = date % 100
            dt = datetime.datetime(year, month, day, hour=hour, minute=minute)
            dt64 = np.datetime64(dt)
            time[i] = dt64
        dataset = dataset.rename(dict(t='time'))
        dataset = dataset.drop(['DATE', 'HOUR', 'MIN'])
        dataset = dataset.assign_coords(time=xr.DataArray(time,
                                                          dims='time',
                                                          attrs=dict(long_name='time',
                                                                     standard_name='time',
                                                                     units='seconds since 1970-01-01'),
                                                          encoding=dict(units='seconds since 1970-01-01',
                                                                        calendar='standard')))
        return dataset


def init_plugin():
    """ Register a DatasetIO object: SnapOlciHighrocL2NetcdfInputProcessor() """
    register_input_processor(RbinsSeviriHighrocSceneInputProcessor())
    register_input_processor(RbinsSeviriHighrocDailyInputProcessor())
