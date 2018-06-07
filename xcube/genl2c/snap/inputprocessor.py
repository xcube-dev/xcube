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

import xarray as xr

from xcube.io import get_default_dataset_io_registry
from .mask import mask_dataset
from ..inputprocessor import InputProcessor


class SnapInputProcessor(InputProcessor, metaclass=ABCMeta):
    """
    Input processor for SNAP L2 NetCDF inputs.
    """

    def __init__(self, expr_pattern=None):
        self.expr_pattern = expr_pattern

    def read(self, input_file: str, **kwargs) -> xr.Dataset:
        """ Read SNAP L2 NetCDF inputs. """
        return xr.open_dataset(input_file, decode_cf=True, decode_coords=True, decode_times=False)

    @classmethod
    def pre_reproject(cls, dataset: xr.Dataset) -> xr.Dataset:
        """ Do any pre-processing before reprojection. """
        masked_dataset, _ = mask_dataset(dataset,
                                         expr_pattern='({expr}) AND !quality_flags.land',
                                         errors='raise')
        return masked_dataset


# noinspection PyAbstractClass
class SnapOlciHighrocL2InputProcessor(SnapInputProcessor):
    """
    Input processor for SNAP Sentinel-3 OLCI HIGHROC Level-2 NetCDF inputs.
    """

    def __init__(self):
        super().__init__(expr_pattern='({expr}) AND !quality_flags.land')

    @property
    def name(self) -> str:
        return 'snap-olci-highroc-l2'

    @property
    def description(self) -> str:
        return 'SNAP Sentinel-3 OLCI HIGHROC Level-2 NetCDF inputs'

    @property
    def ext(self) -> str:
        return 'nc'


def init_plugin():
    ds_io_registry = get_default_dataset_io_registry()
    ds_io_registry.register(SnapOlciHighrocL2InputProcessor())
