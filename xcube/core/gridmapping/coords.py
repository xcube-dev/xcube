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

import abc
from typing import Tuple, Union

import numpy as np
import pyproj
import xarray as xr

from xcube.util.assertions import assert_condition
from xcube.util.assertions import assert_instance
from .base import GridMapping
from .helpers import _parse_int_pair
from .helpers import _to_int_or_float
from .helpers import to_lon_360


class CoordsGridMapping(GridMapping, abc.ABC):
    """Grid mapping constructed from 1D/2D coordinate variables and a CRS."""

    def __init__(self,
                 /,
                 x_coords: xr.DataArray,
                 y_coords: xr.DataArray,
                 **kwargs):
        self._x_coords = x_coords
        self._y_coords = y_coords
        self._xy_coords = None
        super().__init__(**kwargs)

    @property
    def xy_coords(self) -> xr.DataArray:
        if self._xy_coords is None:
            self._xy_coords = self.new_xy_coords()
        return self._xy_coords

    @property
    def xy_coords_chunks(self) -> Tuple[int, int, int]:
        return 2, self.tile_height, self.tile_width

    @abc.abstractmethod
    def new_xy_coords(self) -> xr.DataArray:
        """Create new coordinate array of shape (2, height, width)."""


class Coords1DGridMapping(CoordsGridMapping):
    """Grid mapping constructed from 1D coordinate variables and a CRS."""

    def new_xy_coords(self) -> xr.DataArray:
        x, y = xr.broadcast(self._y_coords, self._x_coords)
        return xr.concat([x, y], dim='coord').chunk(self.xy_coords_chunks)


class Coords2DGridMapping(CoordsGridMapping):
    """Grid mapping constructed from 2D coordinate variables and a CRS."""

    def new_xy_coords(self) -> xr.DataArray:
        return xr.concat([self._x_coords, self._y_coords], dim='coord').chunk(self.xy_coords_chunks)


def from_coords(x_coords: xr.DataArray,
                y_coords: xr.DataArray,
                crs: pyproj.crs.CRS,
                *,
                tile_size: Union[int, Tuple[int, int]] = None) -> GridMapping:
    assert_instance(x_coords, xr.DataArray)
    assert_instance(y_coords, xr.DataArray)
    assert_condition(x_coords.ndim in (1, 2),
                     'x and y must be either 1D or 2D')
    assert_instance(crs, pyproj.crs.CRS)

    tile_size = _parse_int_pair(tile_size, default=None)
    is_lon_360 = None
    if crs.is_geographic:
        is_lon_360 = np.any(x_coords > 180)

    if x_coords.ndim == 1:
        cls = Coords1DGridMapping
        size = x_coords.size, y_coords.size

        x_dim, y_dim = x_coords.dims[0], y_coords.dims[0]

        x_diff = _abs_no_zero(x_coords.diff(dim=x_dim))
        y_diff = _abs_no_zero(y_coords.diff(dim=y_dim))

        if not is_lon_360 and crs.is_geographic:
            is_anti_meridian_crossed = np.any(np.nanmax(x_diff) > 180)
            if is_anti_meridian_crossed:
                x_coords = to_lon_360(x_coords)
                x_diff = _abs_no_zero(x_coords.diff(dim=x_dim))
                is_lon_360 = True

        x_res, y_res = x_diff[0], y_diff[0]
        is_regular = np.allclose(x_diff, x_res) and np.allclose(y_diff, y_res)
        if not is_regular:
            x_res, y_res = np.nanmedian(x_diff), np.nanmedian(y_diff)

        if tile_size is None \
                and x_coords.chunks is not None \
                and y_coords.chunks is not None:
            tile_size = max(0, *x_coords.chunks[0]), max(0, *y_coords.chunks[0])

        # Guess j axis direction
        is_j_axis_up = bool(y_coords[0] < y_coords[-1])

    else:
        assert_condition(x_coords.dims == y_coords.dims,
                         'dimensions of x and y must be equal')

        cls = Coords2DGridMapping

        height, width = x_coords.shape
        size = width, height

        dim_y, dim_x = x_coords.dims
        x_x_diff = _abs_no_zero(x_coords.diff(dim=dim_x))
        x_y_diff = _abs_no_zero(x_coords.diff(dim=dim_y))
        y_x_diff = _abs_no_zero(y_coords.diff(dim=dim_x))
        y_y_diff = _abs_no_zero(y_coords.diff(dim=dim_y))

        if not is_lon_360 and crs.is_geographic:
            is_anti_meridian_crossed = np.any(np.nanmax(x_x_diff) > 180) or \
                                       np.any(np.nanmax(x_y_diff) > 180)
            if is_anti_meridian_crossed:
                x_coords = to_lon_360(x_coords)
                x_x_diff = _abs_no_zero(x_coords.diff(dim=dim_x))
                x_y_diff = _abs_no_zero(x_coords.diff(dim=dim_y))
                is_lon_360 = True

        if np.all(np.isnan(x_y_diff)) and np.all(np.isnan(y_x_diff)):
            x_res = x_x_diff[0, 0]
            y_res = y_y_diff[0, 0]
            is_regular = np.allclose(x_x_diff[0, :], x_res) and \
                         np.allclose(x_x_diff[-1, :], x_res) and \
                         np.allclose(y_y_diff[:, 0], y_res) and \
                         np.allclose(y_y_diff[:, -1], y_res)
        else:
            x_res = min(np.nanmean(x_x_diff), np.nanmean(x_y_diff))
            y_res = min(np.nanmean(x_x_diff), np.nanmean(x_y_diff))
            is_regular = False

        if tile_size is None and x_coords.chunks is not None:
            j_chunks, i_chunks = x_coords.chunks
            tile_size = max(0, *i_chunks), max(0, *j_chunks)

        if tile_size is not None:
            tile_width, tile_height = tile_size
            x_coords = x_coords.chunk((tile_height, tile_width))
            y_coords = y_coords.chunk((tile_height, tile_width))

        # Guess j axis direction
        is_j_axis_up = np.all(y_coords[0, :] < y_coords[-1, :]) or None

    x_res, y_res = _to_int_or_float(x_res), _to_int_or_float(y_res)
    x_res_05, y_res_05 = x_res / 2, y_res / 2
    x_min = _to_int_or_float(x_coords.min() - x_res_05)
    y_min = _to_int_or_float(y_coords.min() - y_res_05)
    x_max = _to_int_or_float(x_coords.max() + x_res_05)
    y_max = _to_int_or_float(y_coords.max() + y_res_05)

    return cls(x_coords=x_coords,
               y_coords=y_coords,
               crs=crs,
               size=size,
               tile_size=tile_size,
               xy_bbox=(x_min, y_min, x_max, y_max),
               xy_res=(x_res, y_res),
               is_regular=is_regular,
               is_lon_360=is_lon_360,
               is_j_axis_up=is_j_axis_up)


def _abs_no_zero(array: xr.DataArray):
    array = np.absolute(array)
    return np.where(np.isclose(array, 0), np.nan, array)
