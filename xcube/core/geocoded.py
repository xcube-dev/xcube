import math

import numba as nb
import numpy as np
import xarray as xr


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
              dst_res: float):
    src_width = src_values.shape[-1]
    src_height = src_values.shape[-2]

    dst_width = dst_values.shape[-1]
    dst_height = dst_values.shape[-2]

    dst_px = np.zeros(4, dtype=src_x.dtype)
    dst_py = np.zeros(4, dtype=src_y.dtype)

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

            dst_pi_f = (dst_px - dst_x0) / dst_res
            dst_pj_f = (dst_py - dst_y0) / dst_res

            dst_pi = np.floor(dst_pi_f).astype(np.int64)
            dst_pj = np.floor(dst_pj_f).astype(np.int64)

            # dst_pi_f -= dst_pi
            # dst_pj_f -= dst_pj

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
                        if u >= 0.0 and v >= 0.0 and u + v <= 1.0:
                            src_i = src_i0 if u < 0.5 else src_i1
                            src_j = src_j0 if v < 0.5 else src_j1
                    if src_i == -1 and det_b != 0.0:
                        u = _fu(dst_x, dst_y, dst_px3, dst_py3, dst_px1, dst_py1) / det_b
                        v = _fv(dst_x, dst_y, dst_px3, dst_py3, dst_px2, dst_py2) / det_b
                        if u >= 0.0 and v >= 0.0 and u + v <= 1.0:
                            src_i = src_i1 if u < 0.5 else src_i0
                            src_j = src_j1 if v < 0.5 else src_j0
                    if src_i != -1:
                        dst_values[dst_j, dst_i] = src_values[src_j, src_i]


def compute_output_geom(src_ds: xr.Dataset, denom_x: int = 1, denom_y: int = 1, delta: float = 1e-10):
    lon_min = float(src_ds.lon.min())
    lon_max = float(src_ds.lon.max())
    lat_min = float(src_ds.lat.min())
    lat_max = float(src_ds.lat.max())
    lon_x_diff = abs(src_ds.lon.diff(dim='x'))
    lon_y_diff = abs(src_ds.lon.diff(dim='y'))
    lat_x_diff = abs(src_ds.lat.diff(dim='x'))
    lat_y_diff = abs(src_ds.lat.diff(dim='y'))
    lon_x_res = lon_x_diff.where(lon_x_diff > delta).min()
    lon_y_res = lon_y_diff.where(lon_y_diff > delta).min()
    lat_x_res = lat_x_diff.where(lat_x_diff > delta).min()
    lat_y_res = lat_y_diff.where(lat_y_diff > delta).min()
    lon_res = float(min(lon_x_res, lon_y_res))
    lat_res = float(min(lat_x_res, lat_y_res))
    res = min(lon_res, lat_res)
    width = 1 + math.floor((lon_max - lon_min) / res)
    height = 1 + math.floor((lat_max - lat_min) / res)
    return denom_x * ((width + denom_x - 1) // denom_x), \
           denom_y * ((height + denom_y - 1) // denom_y), \
           lon_min, lat_min, res
