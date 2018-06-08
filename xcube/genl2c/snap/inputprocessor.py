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

from abc import ABCMeta
from typing import Optional

import xarray as xr

from .mask import mask_dataset
from .vectorize import vectorize_wavebands
from ..inputprocessor import InputProcessor, InputInfo
from ...constants import CRS_WKT_EPSG_4326
from ...io import get_default_dataset_io_registry


class SnapNetcdfInputProcessor(InputProcessor, metaclass=ABCMeta):
    """
    Input processor for SNAP L2 NetCDF inputs.
    """

    @property
    def ext(self) -> str:
        return 'nc'

    @property
    def extra_expr_pattern(self) -> Optional[str]:
        return None

    @property
    def input_info(self) -> InputInfo:
        return InputInfo(xy_var_names=('lon', 'lat'),
                         xy_tp_var_names=('TP_longitude', 'TP_latitude'),
                         xy_crs=CRS_WKT_EPSG_4326,
                         time_range_attr_names=('start_date', 'stop_date'))

    def read(self, input_file: str, **kwargs) -> xr.Dataset:
        """ Read SNAP L2 NetCDF inputs. """
        return xr.open_dataset(input_file, decode_cf=True, decode_coords=True, decode_times=False)

    def pre_reproject(self, dataset: xr.Dataset) -> xr.Dataset:
        """ Do any pre-processing before reprojection. """
        masked_dataset, _ = mask_dataset(dataset,
                                         expr_pattern=self.extra_expr_pattern,
                                         errors='raise')
        return masked_dataset

    def post_reproject(self, dataset: xr.Dataset) -> xr.Dataset:
        return vectorize_wavebands(dataset)


# noinspection PyAbstractClass
class SnapOlciHighrocL2NetcdfInputProcessor(SnapNetcdfInputProcessor):
    """
    Input processor for SNAP Sentinel-3 OLCI HIGHROC Level-2 NetCDF inputs.
    """

    @property
    def name(self) -> str:
        return 'snap-olci-highroc-l2'

    @property
    def description(self) -> str:
        return 'SNAP Sentinel-3 OLCI HIGHROC Level-2 NetCDF inputs'

    @property
    def extra_expr_pattern(self) -> Optional[str]:
        return '({expr}) AND !quality_flags.land'


def init_plugin():
    """ Register a DatasetIO object: SnapOlciHighrocL2NetcdfInputProcessor() """
    ds_io_registry = get_default_dataset_io_registry()
    ds_io_registry.register(SnapOlciHighrocL2NetcdfInputProcessor())
