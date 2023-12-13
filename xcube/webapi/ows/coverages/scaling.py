# The MIT License (MIT)
# Copyright (c) 2023 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
from typing import Optional

import pyproj
import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.server.api import ApiError
from xcube.webapi.ows.coverages.util import get_h_dim, get_v_dim
from xcube.webapi.ows.coverages.request import CoverageRequest


class CoverageScaling:
    _scale: Optional[tuple[float, float]] = None
    _final_size: Optional[tuple[int, int]] = None
    _initial_size: tuple[int, int]
    _crs: pyproj.CRS
    _x_name: str
    _y_name: str

    def __init__(
        self, request: CoverageRequest, crs: pyproj.CRS, ds: xr.Dataset
    ):
        h_dim = get_h_dim(ds)
        v_dim = get_v_dim(ds)
        for d in h_dim, v_dim:
            size = ds.dims[d]
            if size == 0:
                # Requirement 8C currently specifies a 204 rather than 404 here,
                # but spec will soon be updated to allow 404 as an alternative.
                # (J. Jacovella-St-Louis, pers. comm., 2023-11-27).
                raise ApiError.NotFound(
                    f'Requested coverage contains no data: {d} has zero size.'
                )
        self._initial_size = ds.dims[h_dim], ds.dims[v_dim]

        self._crs = crs
        self._y_name = self.get_axis_from_crs(
            {'lat', 'latitude', 'geodetic latitude', 'n', 'north', 'y'}
        )
        self._x_name = self.get_axis_from_crs(
            {'longitude', 'geodetic longitude', 'lon', 'e', 'east', 'x'}
        )

        # The standard doesn't define behaviour when multiple scale
        # parameters are given. We choose to handle one and ignore the
        # others in such cases.
        if request.scale_factor is not None:
            self._scale = request.scale_factor, request.scale_factor
        elif request.scale_axes is not None:
            self._scale = self._get_xy_values(request.scale_axes)
        elif request.scale_size is not None:
            # The standard allows floats for "scale-size" but mandates:
            # "The returned coverage SHALL contain exactly the specified number
            # of sample values". We can't return a fractional number of sample
            # values, so truncate to int here.
            x, y = self._get_xy_values(request.scale_size)
            self._final_size = int(x), int(y)
        else:
            # The standard doesn't mandate this as a default; in future, we
            # might choose to downscale automatically if a large coverage
            # is requested without an explicit scaling parameter.
            self._scale = (1, 1)

    @property
    def scale(self) -> tuple[float, float]:
        if self._scale is not None:
            return self._scale
        else:
            x_initial, y_initial = self._initial_size
            x_final, y_final = self._final_size
            return x_initial / x_final, y_initial / y_final

    @property
    def size(self) -> tuple[float, float]:
        if self._final_size is not None:
            return self._final_size
        else:
            x_initial, y_initial = self._initial_size
            x_scale, y_scale = self._scale
            return x_initial / x_scale, y_initial / y_scale

    def _get_xy_values(
        self, axis_to_value: dict[str, float]
    ) -> tuple[float, float]:
        x, y = None, None
        for axis in axis_to_value:
            if axis.lower()[:3] in ['x', 'e', 'eas', 'lon', self._x_name]:
                x = axis_to_value[axis]
            if axis.lower()[:3] in ['y', 'n', 'nor', 'lat', self._y_name]:
                y = axis_to_value[axis]
        return x, y

    def get_axis_from_crs(self, valid_identifiers: set[str]):
        for axis in self._crs.axis_info:
            if not hasattr(axis, 'abbrev'):
                continue
            identifiers = set(
                map(
                    lambda attr: getattr(axis, attr, '').lower(),
                    ['name', 'abbrev'],
                )
            )
            if identifiers & valid_identifiers:
                return axis.abbrev
        return None

    def apply(self, gm: GridMapping):
        if self.scale == (1, 1):
            return gm
        else:
            regular = gm.to_regular()
            source = regular.size
            return \
                regular.scale((self.size[0] / source[0], self.size[1] / source[1]))
