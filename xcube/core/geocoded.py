import collections
import math
from typing import Sequence, Tuple

import numba as nb
import numpy as np
import xarray as xr


class ImageGeom(collections.namedtuple('ImageGeometry', ['width', 'height', 'x_min', 'y_min', 'res'])):
    @property
    def is_crossing_antimeridian(self):
        return self.x_min + self.width * self.res > 180.0


@nb.jit(nopython=True, inline='always')
def _fdet(px0: float, py0: float, px1: float, py1: float, px2: float, py2: float) -> float:
    return (px0 - px1) * (py0 - py2) - (px0 - px2) * (py0 - py1)


@nb.jit(nopython=True, inline='always')
def _fu(px: float, py: float, px0: float, py0: float, px2: float, py2: float) -> float:
    return (px0 - px) * (py0 - py2) - (py0 - py) * (px0 - px2)


@nb.jit(nopython=True, inline='always')
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

            dst_px[0] = dst_px0 = src_x[src_j0, src_i0]
            dst_px[1] = dst_px1 = src_x[src_j0, src_i1]
            dst_px[2] = dst_px2 = src_x[src_j1, src_i0]
            dst_px[3] = dst_px3 = src_x[src_j1, src_i1]

            dst_py[0] = dst_py0 = src_y[src_j0, src_i0]
            dst_py[1] = dst_py1 = src_y[src_j0, src_i1]
            dst_py[2] = dst_py2 = src_y[src_j1, src_i0]
            dst_py[3] = dst_py3 = src_y[src_j1, src_i1]

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

            det_a = _fdet(dst_px0, dst_py0, dst_px1, dst_py1, dst_px2, dst_py2)
            det_b = _fdet(dst_px3, dst_py3, dst_px2, dst_py2, dst_px1, dst_py1)
            for dst_j in range(dst_j_min, dst_j_max + 1):
                dst_y = dst_y0 + dst_j * dst_res
                for dst_i in range(dst_i_min, dst_i_max + 1):
                    dst_x = dst_x0 + dst_i * dst_res

                    # TODO: test which Ps are valid

                    src_i = -1
                    src_j = -1
                    if det_a != 0.0:
                        u = _fu(dst_x, dst_y, dst_px0, dst_py0, dst_px2, dst_py2) / det_a
                        v = _fv(dst_x, dst_y, dst_px0, dst_py0, dst_px1, dst_py1) / det_a
                        if u >= u_min and v >= v_min and u + v <= uv_max:
                            src_i = src_i0 if u < 0.5 else src_i1
                            src_j = src_j0 if v < 0.5 else src_j1
                    if src_i == -1 and det_b != 0.0:
                        u = _fu(dst_x, dst_y, dst_px3, dst_py3, dst_px1, dst_py1) / det_b
                        v = _fv(dst_x, dst_y, dst_px3, dst_py3, dst_px2, dst_py2) / det_b
                        if u >= u_min and v >= v_min and u + v <= uv_max:
                            src_i = src_i1 if u < 0.5 else src_i0
                            src_j = src_j1 if v < 0.5 else src_j0
                    if src_i != -1:
                        dst_values[dst_j, dst_i] = src_values[src_j, src_i]


def compute_output_geom(src_ds: xr.Dataset,
                        x_name: str = 'lon',
                        y_name: str = 'lat',
                        oversampling: float = 1.0,
                        denom_x: int = 1,
                        denom_y: int = 1,
                        delta: float = 1e-10) -> ImageGeom:
    src_x, src_y, normalized_lon = _get_2d_coords(src_ds, x_name=x_name, y_name=y_name)
    src_x_x_diff = src_x.diff(dim='x')
    src_x_y_diff = src_x.diff(dim='y')
    src_y_x_diff = src_y.diff(dim='x')
    src_y_y_diff = src_y.diff(dim='y')
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
    dim_x = lon_var.dims[-1]
    dim_y = lon_var.dims[-2]
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
                      delta: float = 1e-3):
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
        output_geom = compute_output_geom(src_ds)
    dst_width, dst_height, dst_x_min, dst_y_min, dst_res = output_geom

    dims = (y_name, x_name)
    # TODO: add bnds vars too
    x_var = xr.DataArray(np.linspace(dst_x_min, dst_x_min + (dst_width - 1) * dst_res,
                                     num=dst_width, dtype=np.float64), dims=x_name)
    y_var = xr.DataArray(np.linspace(dst_y_min, dst_y_min + (dst_height - 1) * dst_res,
                                     num=dst_height, dtype=np.float64), dims=y_name)
    coords = {
        x_name: _denormalize_lon(x_var) if normalized_lon else x_var,
        y_name: y_var
    }

    src_x_values = src_x.values
    src_y_values = src_y.values

    dst_vars = dict()
    for src_var in src_vars:
        dst_var_values = np.full((dst_height, dst_width), np.nan, dtype=src_var.dtype)
        reproject(src_var.values,
                  src_x_values,
                  src_y_values,
                  dst_var_values,
                  dst_x_min,
                  dst_y_min,
                  dst_res,
                  delta=delta)
        dst_vars[src_var.name] = xr.DataArray(dst_var_values, dims=dims, coords=coords, attrs=src_var.attrs)

    return xr.Dataset(dst_vars, attrs=src_ds.attrs)
