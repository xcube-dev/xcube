# The MIT License (MIT)
# Copyright (c) 2018 by the xcube development team and contributors
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

from abc import ABCMeta, abstractmethod
from typing import Tuple

import xarray as xr

from xcube.io import DatasetIO


class InputInfo:
    """
    Characterize input datasets so we can reproject.

    :param xy_var_names: Name of variable providing the spatial x-coordinates,
           e.g. ('lon', 'lat')
    :param xy_tp_var_names: Name of tie-point variable providing the spatial y-coordinates,
           e.g. ('TP_longitude', 'TP_latitude')
    :param xy_crs: Spatial reference system, e.g. 'EPSG:4326'
    :param time_var_name: Name of variable providing time coordinates
    :param time_range_attr_names: Name of dataset attribute providing the first observation time,
           e.g. ('start_date', 'stop_date')
    :param time_format: Time format pattern, e.g. '%Y-%m-%d %H:%M%:%S'.
           None means auto-detected, e.g. in SNAP NetCDF we have values like "14-APR-2017 10:27:50.183264"
           that can be parsed automatically.
    """

    def __init__(self,
                 xy_var_names: Tuple[str, str],
                 xy_tp_var_names: Tuple[str, str] = None,
                 xy_crs: str = None,
                 time_var_name: str = None,
                 time_range_attr_names: Tuple[str, str] = None,
                 time_format: str = None):
        self.xy_var_names = xy_var_names
        self.xy_tp_var_names = xy_tp_var_names
        self.xy_crs = xy_crs
        self.time_var_name = time_var_name
        self.time_range_attr_names = time_range_attr_names
        self.time_format = time_format


class InputProcessor(DatasetIO, metaclass=ABCMeta):
    """
    Read and process inputs for the genl2c tool.
    """

    @property
    def modes(self):
        return {'r'}

    @property
    @abstractmethod
    def input_info(self) -> InputInfo:
        """ Information about special fields in input datasets. """
        pass

    def pre_reproject(self, dataset: xr.Dataset) -> xr.Dataset:
        """ Do any pre-processing before reprojection. """
        return dataset

    # noinspection PyMethodMayBeStatic
    def post_reproject(self, dataset: xr.Dataset) -> xr.Dataset:
        """ Do any pre-processing before reprojection. """
        return dataset

    def write(self, dataset: xr.Dataset, output_path: str, **kwargs):
        raise NotImplementedError()

    def append(self, dataset: xr.Dataset, output_path: str, **kwargs):
        raise NotImplementedError()
