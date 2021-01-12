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

from typing import Tuple, Union

import dask.array as da
import numpy as np
import pyproj
import xarray as xr

from xcube.util.assertions import assert_instance
from xcube.util.assertions import assert_condition
from .base import GridMapping
from .helpers import _parse_int_pair
from .helpers import _parse_number_pair
from .helpers import _to_int_or_float


class RegularGridMapping(GridMapping):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._xy_coords = None

    @property
    def xy_coords(self) -> xr.DataArray:
        """
        The x,y coordinates as data array of shape (2, height, width).
        Coordinates are given in units of the CRS.
        """
        if self._xy_coords is None:
            self._assert_regular()
            x_res_05, y_res_05 = self.x_res / 2, self.y_res / 2
            x1, x2 = self.x_min + x_res_05, self.x_max - x_res_05
            y1, y2 = self.y_min + y_res_05, self.y_max - y_res_05
            if not self.is_j_axis_up:
                y1, y2 = y2, y1
            x_name, y_name = ('lon', 'lat') if self.crs.is_geographic else ('x', 'y')
            shape = self.height, self.width
            chunks = self.tile_height, self.tile_width
            x_coords = da.broadcast_to(np.linspace(x1, x2, self.width), shape=shape)
            y_coords = da.broadcast_to(np.linspace(y1, y2, self.height), shape=shape)
            xy_coords = da.stack([x_coords, y_coords]).rechunk((2, *chunks))
            self._xy_coords = xr.DataArray(xy_coords,
                                           dims=('coord', y_name, x_name),
                                           name='xy_coords')

        return self._xy_coords


def from_min_res(size: Union[int, Tuple[int, int]],
                 xy_min: Tuple[float, float],
                 xy_res: Union[float, Tuple[float, float]],
                 crs: pyproj.crs.CRS,
                 *,
                 tile_size: Union[int, Tuple[int, int]] = None,
                 is_j_axis_up: bool = False):
    width, height = _parse_int_pair(size, name='size')
    assert_condition(width > 1 and height > 1, 'invalid size')

    x_min, y_min = _parse_number_pair(xy_min, name='xy_min')

    x_res, y_res = _parse_number_pair(xy_res, name='xy_res')
    assert_condition(x_res > 0 and y_res > 0, 'invalid xy_res')

    assert_instance(crs, pyproj.crs.CRS, name='crs')
    assert_instance(is_j_axis_up, bool, name='is_j_axis_up')

    x_min = _to_int_or_float(x_min)
    y_min = _to_int_or_float(y_min)
    x_max = _to_int_or_float(x_min + x_res * width)
    y_max = _to_int_or_float(y_min + y_res * height)

    if crs.is_geographic:
        if y_min < -90:
            raise ValueError('invalid y_min')
        if y_max > 90:
            raise ValueError('invalid size, y_min combination')

    return RegularGridMapping(crs=crs,
                              size=(width, height),
                              tile_size=tile_size,
                              xy_bbox=(x_min, y_min, x_max, y_max),
                              xy_res=(x_res, y_res),
                              is_regular=True,
                              is_lon_360=x_max > 180,
                              is_j_axis_up=is_j_axis_up)
