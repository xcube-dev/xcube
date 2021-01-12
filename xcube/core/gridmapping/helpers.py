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

import math
from typing import Any
from typing import Tuple, Optional, Union

import affine
import numpy as np
import xarray as xr

from xcube.util.assertions import assert_given
from xcube.util.undefined import UNDEFINED

Number = Union[int, float]
AffineTransformMatrix = Tuple[Tuple[Number, Number, Number],
                              Tuple[Number, Number, Number]]


def _to_int_or_float(x: Number) -> Number:
    """
    If x is an int or is close to an int return it as int otherwise as float.
    Helps avoiding errors introduced by inaccurate floating point ops.
    """
    if isinstance(x, int):
        return x
    xi = int(x)
    xf = float(x)
    return xi if math.isclose(xi, xf, abs_tol=1e-10) else xf


def _from_affine(matrix: affine.Affine) -> AffineTransformMatrix:
    return (matrix.a, matrix.b, matrix.c), (matrix.d, matrix.e, matrix.f)


def _to_affine(matrix: AffineTransformMatrix) -> affine.Affine:
    return affine.Affine(*matrix[0], *matrix[1])


def _parse_int_pair(value: Any,
                    name: str = None,
                    default: Optional[Tuple[int, int]] = UNDEFINED) -> Optional[Tuple[int, int]]:
    if isinstance(value, int):
        return value, value
    elif value is not None:
        x, y = value
        return int(x), int(y)
    elif default is not UNDEFINED:
        return default
    else:
        assert_given(name, 'name')
        raise ValueError(f'{name} must be an int or a sequence of two ints')


def _parse_number_pair(value: Any,
                       name: str = None,
                       default: Optional[Tuple[Number, Number]] = UNDEFINED) -> Optional[Tuple[Number, Number]]:
    if isinstance(value, (float, int)):
        x, y = value, value
        return _to_int_or_float(x), _to_int_or_float(y)
    elif value is not None:
        x, y = value
        return _to_int_or_float(x), _to_int_or_float(y)
    elif default is not UNDEFINED:
        return default
    else:
        assert_given(name, 'name')
        raise ValueError(f'{name} must be a number or a sequence of two numbers')


def to_lon_360(lon_var: Union[np.ndarray, xr.DataArray]):
    if isinstance(lon_var, xr.DataArray):
        return lon_var.where(lon_var >= 0.0, lon_var + 360.0)
    else:
        return np.where(lon_var >= 0.0, lon_var, lon_var + 360.0)


def from_lon_360(lon_var: Union[np.ndarray, xr.DataArray]):
    if isinstance(lon_var, xr.DataArray):
        return lon_var.where(lon_var <= 180.0, lon_var - 360.0)
    else:
        return np.where(lon_var <= 180.0, lon_var, lon_var - 360.0)

# def _assert_valid_xy_coords(xy_coords: Any):
#     assert_instance(xy_coords, xr.DataArray, name='xy_coords')
#     assert_condition(xy_coords.ndim == 3
#                      and xy_coords.shape[0] == 2
#                      and xy_coords.shape[1] >= 2
#                      and xy_coords.shape[2] >= 2,
#                      'xy_coords must have dimensions (2, height, width) with height >= 2 and width >= 2')
