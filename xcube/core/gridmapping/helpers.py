# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import math
from fractions import Fraction
from typing import Any
from typing import Tuple, Optional, Union

import affine
import dask.array as da
import numpy as np
import pyproj.crs
import xarray as xr

from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_instance
from xcube.util.assertions import assert_true
from xcube.util.undefined import UNDEFINED

Number = Union[int, float]
AffineTransformMatrix = tuple[
    tuple[Number, Number, Number], tuple[Number, Number, Number]
]


def _to_int_or_float(x: Number) -> Number:
    """If x is an int or is close to an int return it
    as int otherwise as float. Helps avoiding errors
    introduced by inaccurate floating point ops.
    """
    if isinstance(x, int):
        return x
    xf = float(x)
    xi = round(xf)
    return xi if math.isclose(xi, xf, rel_tol=1e-5) else xf


def _from_affine(matrix: affine.Affine) -> AffineTransformMatrix:
    return (matrix.a, matrix.b, matrix.c), (matrix.d, matrix.e, matrix.f)


def _to_affine(matrix: AffineTransformMatrix) -> affine.Affine:
    return affine.Affine(*matrix[0], *matrix[1])


def _normalize_crs(crs: Union[str, pyproj.CRS]) -> pyproj.CRS:
    if isinstance(crs, pyproj.CRS):
        return crs
    assert_instance(crs, str, "crs")
    return pyproj.CRS.from_string(crs)


def _normalize_int_pair(
    value: Any, name: str = None, default: Optional[tuple[int, int]] = UNDEFINED
) -> Optional[tuple[int, int]]:
    if isinstance(value, int):
        return value, value
    elif value is not None:
        x, y = value
        return int(x), int(y)
    elif default != UNDEFINED:
        return default
    else:
        assert_given(name, "name")
        raise ValueError(f"{name} must be an int" f" or a sequence of two ints")


def _normalize_number_pair(
    value: Any, name: str = None, default: Optional[tuple[Number, Number]] = UNDEFINED
) -> Optional[tuple[Number, Number]]:
    if isinstance(value, (float, int)):
        x, y = value, value
        return _to_int_or_float(x), _to_int_or_float(y)
    elif value is not None:
        x, y = value
        return _to_int_or_float(x), _to_int_or_float(y)
    elif default != UNDEFINED:
        return default
    else:
        assert_given(name, "name")
        raise ValueError(f"{name} must be a number" f" or a sequence of two numbers")


def to_lon_360(lon_var: Union[np.ndarray, da.Array, xr.DataArray]):
    if isinstance(lon_var, xr.DataArray):
        return lon_var.where(lon_var >= 0.0, lon_var + 360.0)
    else:
        lon_var = da.asarray(lon_var)
        return da.where(lon_var >= 0.0, lon_var, lon_var + 360.0)


def from_lon_360(lon_var: Union[np.ndarray, da.Array, xr.DataArray]):
    if isinstance(lon_var, xr.DataArray):
        return lon_var.where(lon_var <= 180.0, lon_var - 360.0)
    else:
        lon_var = da.asarray(lon_var)
        return da.where(lon_var <= 180.0, lon_var, lon_var - 360.0)


def _default_xy_var_names(crs: pyproj.crs.CRS) -> tuple[str, str]:
    return ("lon", "lat") if crs.is_geographic else ("x", "y")


def _default_xy_dim_names(crs: pyproj.crs.CRS) -> tuple[str, str]:
    return _default_xy_var_names(crs)


def _assert_valid_xy_names(value: Any, name: str = None):
    assert_instance(value, tuple, name=name)
    assert_true(
        len(value) == 2 and all(value) and value[0] != value[1],
        f'invalid {name or "value"}',
    )


def _assert_valid_xy_coords(xy_coords: Any):
    assert_instance(xy_coords, xr.DataArray, name="xy_coords")
    assert_true(
        xy_coords.ndim == 3
        and xy_coords.shape[0] == 2
        and xy_coords.shape[1] >= 2
        and xy_coords.shape[2] >= 2,
        "xy_coords must have dimensions"
        " (2, height, width) with height >= 2 and width >= 2",
    )


_RESOLUTIONS = {
    10: (1, 0),
    20: (2, 0),
    25: (25, 1),
    50: (5, 0),
    100: (1, -1),
}

_RESOLUTION_SET = {k / 100 for k in _RESOLUTIONS.keys()}


def round_to_fraction(value: float, digits: int = 2, resolution: float = 1) -> Fraction:
    """Round *value* at position given by significant
    *digits* and return result as fraction.

    Args:
        value: The value
        digits: The number of significant digits. Must be an integer >=
            1. Default is 2.
        resolution: The rounding resolution for the least significant
            digit. Must be one of (0.1, 0.2, 0.25, 0.5, 1). Default is
            1.

    Returns:
        The rounded value as fraction.Fraction instance.
    """
    if digits < 1:
        raise ValueError("digits must be a positive integer")
    resolution_key = round(100 * resolution)
    if resolution_key not in _RESOLUTIONS or not math.isclose(
        100 * resolution, resolution_key
    ):
        raise ValueError(f"resolution must be one of {_RESOLUTION_SET}")
    if value == 0:
        return Fraction(0, 1)
    sign = 1
    if value < 0:
        sign = -1
        value = -value
    resolution, resolution_digits = _RESOLUTIONS[resolution_key]
    exponent = math.floor(math.log10(value)) - digits - resolution_digits
    if exponent >= 0:
        magnitude = Fraction(10**exponent, 1)
    else:
        magnitude = Fraction(1, 10**-exponent)
    scaled_value = value / magnitude
    discrete_value = resolution * round(scaled_value / resolution)
    return (sign * discrete_value) * magnitude


def scale_xy_res_and_size(
    xy_res: tuple[float, float], size: tuple[int, int], xy_scale: tuple[float, float]
) -> tuple[tuple[float, float], tuple[int, int]]:
    """Scale given *xy_res* and *size* using *xy_scale*.
    Make sure, size components are not less than 2.
    """
    x_res, y_res = xy_res
    x_scale, y_scale = xy_scale
    w, h = size
    w, h = round(x_scale * w), round(y_scale * h)
    return (
        (x_res / x_scale, y_res / y_scale),
        (w if w >= 2 else 2, h if h >= 2 else 2),
    )
