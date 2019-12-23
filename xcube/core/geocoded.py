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

import collections
import math
from typing import Sequence, Tuple, Optional

import numba as nb
import numpy as np
import xarray as xr


class ImageGeom(collections.namedtuple('ImageGeometry', ['width', 'height', 'x_min', 'y_min', 'res'])):
    @property
    def is_crossing_antimeridian(self):
        return self.x_min + self.width * self.res > 180.0


@nb.jit('float64(float64, float64, float64, float64, float64, float64)',
        nopython=True, inline='always')
def _fdet(px0: float, py0: float, px1: float, py1: float, px2: float, py2: float) -> float:
    return (px0 - px1) * (py0 - py2) - (px0 - px2) * (py0 - py1)


@nb.jit('float64(float64, float64, float64, float64, float64, float64)',
        nopython=True, inline='always')
def _fu(px: float, py: float, px0: float, py0: float, px2: float, py2: float) -> float:
    return (px0 - px) * (py0 - py2) - (py0 - py) * (px0 - px2)


@nb.jit('float64(float64, float64, float64, float64, float64, float64)',
        nopython=True, inline='always')
def _fv(px: float, py: float, px0: float, py0: float, px1: float, py1: float) -> float:
    return (py0 - py) * (px0 - px1) - (px0 - px) * (py0 - py1)


@nb.jit(nopython=True, cache=True)
def reproject(src_values: np.ndarray,
              src_x: np.ndarray,
              src_y: np.ndarray,
              dst_values: np.ndarray,
              dst_x0: float,
              dst_y0: float,
              dst_res: float,
              delta: float = 1e-3):
    src_width = src_values.shape[-1]
    src_height = src_values.shape[-2]

    dst_width = dst_values.shape[-1]
    dst_height = dst_values.shape[-2]

    dst_px = np.zeros(4, dtype=src_x.dtype)
    dst_py = np.zeros(4, dtype=src_y.dtype)

    u_min = v_min = -delta
    uv_max = 1.0 + 2 * delta

    for src_j0 in range(src_height - 1):
        for src_i0 in range(src_width - 1):
            src_i1 = src_i0 + 1
            src_j1 = src_j0 + 1

            dst_px[0] = dst_p0x = src_x[src_j0, src_i0]
            dst_px[1] = dst_p1x = src_x[src_j0, src_i1]
            dst_px[2] = dst_p2x = src_x[src_j1, src_i0]
            dst_px[3] = dst_p3x = src_x[src_j1, src_i1]

            dst_py[0] = dst_p0y = src_y[src_j0, src_i0]
            dst_py[1] = dst_p1y = src_y[src_j0, src_i1]
            dst_py[2] = dst_p2y = src_y[src_j1, src_i0]
            dst_py[3] = dst_p3y = src_y[src_j1, src_i1]

            dst_pi = np.floor((dst_px - dst_x0) / dst_res).astype(np.int64)
            dst_pj = np.floor((dst_py - dst_y0) / dst_res).astype(np.int64)

            dst_i_min = np.min(dst_pi)
            dst_i_max = np.max(dst_pi)
            dst_j_min = np.min(dst_pj)
            dst_j_max = np.max(dst_pj)

            if dst_i_max < 0 \
                    or dst_j_max < 0 \
                    or dst_i_min >= dst_width \
                    or dst_j_min >= dst_height:
                continue

            if dst_i_min < 0:
                dst_i_min = 0

            if dst_i_max >= dst_width:
                dst_i_max = dst_width - 1

            if dst_j_min < 0:
                dst_j_min = 0

            if dst_j_max >= dst_height:
                dst_j_max = dst_height - 1

            # u from p0 right to p1, v from p0 down to p2
            det_a = _fdet(dst_p0x, dst_p0y, dst_p1x, dst_p1y, dst_p2x, dst_p2y)
            # u from p3 left to p2, v from p3 up to p1
            det_b = _fdet(dst_p3x, dst_p3y, dst_p2x, dst_p2y, dst_p1x, dst_p1y)
            for dst_j in range(dst_j_min, dst_j_max + 1):
                dst_y = dst_y0 + dst_j * dst_res
                for dst_i in range(dst_i_min, dst_i_max + 1):
                    dst_x = dst_x0 + dst_i * dst_res

                    # TODO: use two other combinations,
                    #       if one of the dst_px<n>,dst_py<n> pairs is missing.
                    # TODO: allow returning just src_i + u, src_j + v.
                    #       They can later be used to reproject all variables fast.
                    # TODO: allow returning just src_i + u, src_j + v.
                    #       They can later be used to reproject all variables fast.

                    src_i = -1
                    src_j = -1
                    if det_a != 0.0:
                        u = _fu(dst_x, dst_y, dst_p0x, dst_p0y, dst_p2x, dst_p2y) / det_a
                        v = _fv(dst_x, dst_y, dst_p0x, dst_p0y, dst_p1x, dst_p1y) / det_a
                        if u >= u_min and v >= v_min and u + v <= uv_max:
                            src_i = src_i0 if u < 0.5 else src_i1
                            src_j = src_j0 if v < 0.5 else src_j1
                    if src_i == -1 and det_b != 0.0:
                        u = _fu(dst_x, dst_y, dst_p3x, dst_p3y, dst_p1x, dst_p1y) / det_b
                        v = _fv(dst_x, dst_y, dst_p3x, dst_p3y, dst_p2x, dst_p2y) / det_b
                        if u >= u_min and v >= v_min and u + v <= uv_max:
                            src_i = src_i1 if u < 0.5 else src_i0
                            src_j = src_j1 if v < 0.5 else src_j0
                    if src_i != -1:
                        dst_values[..., dst_j, dst_i] = src_values[..., src_j, src_i]


def compute_output_geom(src_ds: xr.Dataset,
                        x_name: str = 'lon',
                        y_name: str = 'lat',
                        oversampling: float = 1.0,
                        denom_x: int = 1,
                        denom_y: int = 1,
                        delta: float = 1e-10) -> ImageGeom:
    src_x, src_y, normalized_lon = _get_2d_coords(src_ds, x_name=x_name, y_name=y_name)
    dim_y, dim_x = src_x.dims
    src_x_x_diff = src_x.diff(dim=dim_x)
    src_x_y_diff = src_x.diff(dim=dim_y)
    src_y_x_diff = src_y.diff(dim=dim_x)
    src_y_y_diff = src_y.diff(dim=dim_y)
    src_x_x_diff_sq = np.square(src_x_x_diff)
    src_x_y_diff_sq = np.square(src_x_y_diff)
    src_y_x_diff_sq = np.square(src_y_x_diff)
    src_y_y_diff_sq = np.square(src_y_y_diff)
    src_x_diff = np.sqrt(src_x_x_diff_sq + src_y_x_diff_sq)
    src_y_diff = np.sqrt(src_x_y_diff_sq + src_y_y_diff_sq)
    src_x_res = float(src_x_diff.where(src_x_diff > delta).min())
    src_y_res = float(src_y_diff.where(src_y_diff > delta).min())
    src_res = min(src_x_res, src_y_res) / (math.sqrt(2.0) * oversampling)
    src_x_min = float(src_x.min())
    src_x_max = float(src_x.max())
    src_y_min = float(src_y.min())
    src_y_max = float(src_y.max())
    dst_width = 1 + math.floor((src_x_max - src_x_min) / src_res)
    dst_height = 1 + math.floor((src_y_max - src_y_min) / src_res)
    return ImageGeom(width=denom_x * ((dst_width + denom_x - 1) // denom_x),
                     height=denom_y * ((dst_height + denom_y - 1) // denom_y),
                     x_min=src_x_min,
                     y_min=src_y_min,
                     res=src_res)


def _get_2d_coord(src_ds: xr.Dataset, name: str) -> xr.DataArray:
    if name not in src_ds:
        raise ValueError(f'missing 2D coordinate variable {name!r}')
    var = src_ds[name]
    if var.ndim != 2:
        raise ValueError(f'coordinate variable {name!r} must have two dimensions')
    return var


def _get_2d_coords(src_ds: xr.Dataset,
                   x_name: str = 'lon',
                   y_name: str = 'lat') -> Tuple[xr.DataArray, xr.DataArray, bool]:
    src_x = _get_2d_coord(src_ds, x_name)
    src_y = _get_2d_coord(src_ds, y_name)

    if src_x.shape != src_y.shape or src_x.dims != src_y.dims:
        raise ValueError(f"coordinate variables {x_name!r} and {y_name!r} must have same shape and dimensions")

    src_width, src_height = src_x.shape
    if src_width < 2 or src_height < 2:
        raise ValueError(f"size in each dimension of {x_name!r} and {y_name!r} must be greater two")

    normalized_lon = False
    if x_name in ('lon', 'long', 'longitude'):
        src_x, normalized_lon = _maybe_normalise_lon(src_x)

    return src_x, src_y, normalized_lon


def _is_2d_var(var: xr.DataArray, coord_var: xr.DataArray) -> bool:
    return var.ndim >= 2 and var.shape[-2:] == coord_var.shape and var.dims[-2:] == coord_var.dims


def _is_crossing_antimeridian(lon_var: xr.DataArray):
    dim_y, dim_x = lon_var.dims
    # noinspection PyTypeChecker
    return abs(lon_var.diff(dim=dim_x)).max() > 180.0 or \
           abs(lon_var.diff(dim=dim_y)).max() > 180.0


def _maybe_normalise_lon(lon_var: xr.DataArray):
    if _is_crossing_antimeridian(lon_var):
        lon_var = _normalize_lon(lon_var)
        if _is_crossing_antimeridian(lon_var):
            raise ValueError('cannot account for longitudial anti-meridian crossing')
        return lon_var, True
    return lon_var, False


def _normalize_lon(lon_var: xr.DataArray):
    return lon_var.where(lon_var >= 0.0, lon_var + 360.0)


def _denormalize_lon(lon_var: xr.DataArray):
    return lon_var.where(lon_var <= 180.0, lon_var - 360.0)


def reproject_dataset(src_ds: xr.Dataset,
                      var_names: Sequence[str] = None,
                      x_name: str = 'lon',
                      y_name: str = 'lat',
                      output_geom: ImageGeom = None,
                      delta: float = 1e-3) -> Optional[xr.Dataset]:
    src_x, src_y, normalized_lon = _get_2d_coords(src_ds, x_name=x_name, y_name=y_name)

    if var_names is None:
        var_names = [var_name for var_name, var in src_ds.data_vars.items()
                     if var_name not in (x_name, y_name) and _is_2d_var(var, src_x)]
    elif isinstance(var_names, str):
        var_names = (var_names,)
    elif len(var_names) == 0:
        raise ValueError(f'empty var_names')

    src_vars = []
    for var_name in var_names:
        src_var = src_ds[var_name]
        if not _is_2d_var(src_var, src_x):
            raise ValueError(
                f"cannot reproject variable {var_name!r} as its shape or dimensions "
                f"do not match those of {x_name!r} and {y_name!r}")
        src_vars.append(src_var)

    if output_geom is None:
        output_geom = compute_output_geom(src_ds, x_name=x_name, y_name=y_name)
        dst_width, dst_height, dst_x_min, dst_y_min, dst_res = output_geom
    else:
        dst_width, dst_height, dst_x_min, dst_y_min, dst_res = output_geom
        dst_x_max = dst_x_min + dst_res * dst_width
        dst_y_max = dst_y_min + dst_res * dst_height
        bbox = np.logical_and(np.logical_and(src_x >= dst_x_min, src_x <= dst_x_max),
                              np.logical_and(src_y >= dst_y_min, src_y <= dst_y_max))
        dim_y, dim_x = src_x.dims
        src_i = src_ds[dim_x].where(bbox)
        src_j = src_ds[dim_y].where(bbox)
        i_min = src_i.min()
        i_max = src_i.max()
        j_min = src_j.min()
        j_max = src_j.max()
        if not np.isfinite(i_min) or not np.isfinite(j_min) \
                or not np.isfinite(i_max) or not np.isfinite(j_max):
            return None
        src_width, src_height = src_x.shape
        i1 = int(i_min)
        if i1 > 0:
            i1 -= 1
        i2 = int(i_max + 1)
        if i2 < src_width:
            i2 += 1
        j1 = int(j_min)
        if j1 > 0:
            j1 -= 1
        j2 = int(j_max + 1)
        if j2 < src_height:
            j2 += 1
        if i1 > 0 or j1 > 0 or i2 < src_width or j2 < src_height:
            print(80 * '_')
            i_slice = slice(i1, i2)
            j_slice = slice(j1, j2)
            dim_y, dim_x = src_x.dims
            indexers = {dim_x: i_slice, dim_y: j_slice}
            src_x = src_x.isel(**indexers)
            src_y = src_y.isel(**indexers)
            src_vars = tuple(src_var.isel(**indexers) for src_var in src_vars)

    x_var = xr.DataArray(np.linspace(dst_x_min, dst_x_min + dst_res * (dst_width - 1),
                                     num=dst_width, dtype=np.float64), dims=x_name)
    y_var = xr.DataArray(np.linspace(dst_y_min, dst_y_min + dst_res * (dst_height - 1),
                                     num=dst_height, dtype=np.float64), dims=y_name)
    coords = {
        x_name: _denormalize_lon(x_var) if normalized_lon else x_var,
        y_name: y_var
    }
    dims = (y_name, x_name)

    src_x_values = src_x.values
    src_y_values = src_y.values

    dst_vars = dict()
    for src_var in src_vars:
        dst_var_shape = src_var.shape[0:-2] + (dst_height, dst_width)
        dst_var_values = np.full(dst_var_shape, np.nan, dtype=src_var.dtype)
        reproject(src_var.values,
                  src_x_values,
                  src_y_values,
                  dst_var_values,
                  dst_x_min,
                  dst_y_min,
                  dst_res,
                  delta=delta)
        dst_var_dims = src_var.dims[0:-2] + dims
        dst_var_coords = dict(src_var.coords)
        dst_var_coords.update(**coords)
        dst_vars[src_var.name] = xr.DataArray(dst_var_values,
                                              dims=dst_var_dims,
                                              coords=dst_var_coords,
                                              attrs=src_var.attrs)

    return xr.Dataset(dst_vars, attrs=src_ds.attrs)
