# The MIT License (MIT)
# Copyright (c) 2020 by the xcube development team and contributors
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

from typing import Sequence, Tuple, Optional, Union, Mapping

import dask.array as da
import numba as nb
import numpy as np
import xarray as xr

from xcube.core.geocoding import GeoCoding
from xcube.core.imgeom import ImageGeom
from xcube.util.dask import ChunkContext
from xcube.util.dask import compute_array_from_func


def rectify_dataset(dataset: xr.Dataset,
                    var_names: Union[str, Sequence[str]] = None,
                    geo_coding: GeoCoding = None,
                    xy_names: Tuple[str, str] = None,
                    output_geom: ImageGeom = None,
                    is_y_axis_inverted: bool = False,
                    tile_size: Union[int, Tuple[int, int]] = None,
                    compute_xy: bool = False,
                    compute_vars: bool = False,
                    uv_delta: float = 1e-3) -> Optional[xr.Dataset]:
    """
    Reproject *dataset* using its per-pixel x,y coordinates or the given *geo_coding*.

    The function expects *dataset* to have either one- or two-dimensional coordinate variables
    that provide spatial x,y coordinates for every data variable with the same spatial dimensions.

    For example, a dataset may comprise variables with spatial dimensions ``var(..., y_dim, x_dim)``, then one
    the function expects coordinates to be provided in two forms:

    1. One-dimensional ``x_var(x_dim)`` and ``y_var(y_dim)`` (coordinate) variables.
    2. Two-dimensional ``x_var(y_dim, x_dim)`` and ``y_var(y_dim, x_dim)`` (coordinate) variables.

    If *output_geom* is given and defines a tile size or *tile_size* is given, and the number of tiles
    is greater than one in the output's x- or y-direction, then the returned dataset will be composed of lazy,
    chunked dask arrays. Otherwise the returned dataset will be composed of ordinary numpy arrays.

    :param dataset: Source dataset.
    :param var_names: Optional variable name or sequence of variable names.
    :param geo_coding: Optional dataset geo-coding.
    :param xy_names: Optional tuple of the x- and y-coordinate variables in *dataset*. Ignored if *geo_coding* is given.
    :param output_geom: Optional output geometry. If not given, output geometry will be computed
        to spatially fit *dataset* and to retain its spatial resolution.
    :param is_y_axis_inverted: Whether the y-axis labels in the output should be in inverse order.
    :param tile_size: Optional tile size for the output.
    :param compute_xy: Compute x,y coordinates and load into memory before the actual rectification process.
        May improve runtime performance at the cost of higher memory consumption.
    :param compute_vars: Compute source variables and load into memory before the actual rectification process.
        May improve runtime performance at the cost of higher memory consumption.
    :param uv_delta: A normalized value that is used to determine whether x,y coordinates in the output are contained
        in the triangles defined by the input x,y coordinates.
        The higher this value, the more inaccurate the rectification will be.
    :return: a reprojected dataset, or None if the requested output does not intersect with *dataset*.
    """
    src_geo_coding = geo_coding if geo_coding is not None else GeoCoding.from_dataset(dataset, xy_names=xy_names)
    src_x, src_y = src_geo_coding.xy
    src_attrs = dict(dataset.attrs)

    if output_geom is None:
        output_geom = ImageGeom.from_dataset(dataset, geo_coding=src_geo_coding)
    else:
        src_bbox = src_geo_coding.ij_bbox(output_geom.xy_bbox, ij_border=1, xy_border=output_geom.xy_res)
        if src_bbox[0] == -1:
            return None
        dataset_subset = select_spatial_subset(dataset, src_bbox, geo_coding=src_geo_coding)
        if dataset_subset is not dataset:
            src_geo_coding = GeoCoding.from_dataset(dataset_subset)
            src_x, src_y = src_geo_coding.x, src_geo_coding.y
            dataset = dataset_subset

    if tile_size is not None:
        output_geom = output_geom.derive(tile_size=tile_size)

    src_vars = select_variables(dataset, var_names, geo_coding=src_geo_coding)

    if compute_xy:
        # This is NOT faster:
        src_x = src_x.compute()
        src_y = src_y.compute()
        src_geo_coding = src_geo_coding.derive(x=src_x, y=src_y)

    is_output_tiled = output_geom.is_tiled
    if not is_output_tiled:
        get_dst_var_array = get_dst_var_array_numpy
    else:
        get_dst_var_array = get_dst_var_array_dask

    dst_dims = src_geo_coding.xy_names[::-1]
    dst_ds_coords = output_geom.coord_vars(xy_names=src_geo_coding.xy_names,
                                           is_lon_normalized=src_geo_coding.is_lon_normalized,
                                           is_y_axis_inverted=is_y_axis_inverted)
    dst_vars = dict()
    for src_var_name, src_var in src_vars.items():
        if compute_vars:
            # This is NOT faster:
            src_var = src_var.compute()

        dst_var_dims = src_var.dims[0:-2] + dst_dims
        dst_var_coords = {d: src_var.coords[d] for d in dst_var_dims if d in src_var.coords}
        dst_var_coords.update({d: dst_ds_coords[d] for d in dst_var_dims if d in dst_ds_coords})
        dst_var_array = get_dst_var_array(src_var,
                                          src_geo_coding,
                                          output_geom,
                                          is_y_axis_inverted,
                                          uv_delta)
        dst_var = xr.DataArray(dst_var_array,
                               dims=dst_var_dims,
                               coords=dst_var_coords,
                               attrs=src_var.attrs)
        dst_vars[src_var_name] = dst_var
    return xr.Dataset(dst_vars, coords=dst_ds_coords, attrs=src_attrs)


def get_dst_var_array_numpy(src_var: xr.DataArray,
                            src_geo_coding: GeoCoding,
                            output_geom: ImageGeom,
                            is_dst_y_axis_inverted: bool,
                            uv_delta: float) -> np.ndarray:
    dst_width = output_geom.width
    dst_height = output_geom.height
    dst_var_shape = src_var.shape[0:-2] + (dst_height, dst_width)
    dst_var_array = np.full(dst_var_shape, np.nan, dtype=src_var.dtype)
    dst_x_min = output_geom.x_min
    dst_y_min = output_geom.y_min
    dst_xy_res = output_geom.xy_res
    _reproject(src_var.values,
               src_geo_coding.x.values,
               src_geo_coding.y.values,
               dst_var_array,
               dst_x_min,
               dst_y_min,
               dst_xy_res,
               uv_delta=uv_delta)
    if is_dst_y_axis_inverted:
        dst_var_array = dst_var_array[..., ::-1, :]
    return dst_var_array


def get_dst_var_array_dask(src_var: xr.DataArray,
                           src_geo_coding: GeoCoding,
                           output_geom: ImageGeom,
                           is_dst_y_axis_inverted: bool,
                           uv_delta: float) -> da.Array:
    dst_width = output_geom.width
    dst_height = output_geom.height
    dst_tile_width = output_geom.tile_width
    dst_tile_height = output_geom.tile_height
    dst_var_shape = src_var.shape[0:-2] + (dst_height, dst_width)
    dst_var_chunks = src_var.shape[0:-2] + (dst_tile_height, dst_tile_width)

    dst_x_min = output_geom.x_min
    dst_y_min = output_geom.y_min
    dst_xy_res = output_geom.xy_res

    dst_xy_bboxes = output_geom.xy_bboxes
    src_ij_bboxes = src_geo_coding.ij_bboxes(dst_xy_bboxes, xy_border=dst_xy_res, ij_border=1)

    return compute_array_from_func(_rectify_func_dask,
                                   dst_var_shape,
                                   dst_var_chunks,
                                   src_var.dtype,
                                   args=(
                                       src_var,
                                       src_geo_coding.x,
                                       src_geo_coding.y,
                                       src_ij_bboxes,
                                       dst_x_min,
                                       dst_y_min,
                                       dst_xy_res,
                                       is_dst_y_axis_inverted,
                                       uv_delta
                                   ),
                                   name=src_var.name,
                                   )


def _rectify_func_dask(context: ChunkContext,
                       src_var: xr.DataArray,
                       src_x: xr.DataArray,
                       src_y: xr.DataArray,
                       src_ij_bboxes: np.ndarray,
                       dst_x_min: float,
                       dst_y_min: float,
                       dst_xy_res: float,
                       is_dst_y_axis_inverted: bool,
                       uv_delta: float) -> np.ndarray:
    dst_block = np.full(src_var.shape[:-2] + context.chunk_shape, np.nan, dtype=context.dtype)
    dst_y_slice, dst_x_slice = context.chunk_slices
    src_ij_bbox = src_ij_bboxes[context.chunk_id]
    src_i_min, src_j_min, src_i_max, src_j_max = src_ij_bbox
    if src_i_min == -1:
        return dst_block
    src_i_slice = slice(src_i_min, src_i_max + 1)
    src_j_slice = slice(src_j_min, src_j_max + 1)
    # This is NOT faster:
    # t1 = time.perf_counter()
    # src_indexers = {src_x_dim: src_i_slice, src_y_dim: src_j_slice}
    # src_var_values = src_var.isel(**src_indexers).values
    # src_x_values = src_x.isel(**src_indexers).values
    # src_y_values = src_y.isel(**src_indexers).values
    # t2 = time.perf_counter()
    # t1 = time.perf_counter()
    src_var_values = src_var[..., src_j_slice, src_i_slice].values
    src_x_values = src_x[src_j_slice, src_i_slice].values
    src_y_values = src_y[src_j_slice, src_i_slice].values
    # t2 = time.perf_counter()
    _reproject(src_var_values,
               src_x_values,
               src_y_values,
               dst_block,
               dst_x_min + dst_x_slice.start * dst_xy_res,
               dst_y_min + dst_y_slice.start * dst_xy_res,
               dst_xy_res,
               uv_delta)
    # t3 = time.perf_counter()
    # print(f'target chunk {context.name}-{context.chunk_index}, shape {context.chunk_shape} '
    #       f'for source shape {src_i_max - src_i_min + 1, src_j_max - src_j_min + 1} '
    #       f'took {_millis(t2 - t1)}, {_millis(t3 - t2)} milliseconds, total {_millis(t3 - t1)}')
    if is_dst_y_axis_inverted:
        dst_block = dst_block[..., ::-1, :]
    return dst_block


def select_variables(dataset,
                     var_names: Union[str, Sequence[str]] = None,
                     geo_coding: GeoCoding = None,
                     xy_names: Tuple[str, str] = None) -> Mapping[str, xr.DataArray]:
    """
    Select variables from *dataset*.

    :param dataset: Source dataset.
    :param var_names: Optional variable name or sequence of variable names.
    :param geo_coding: Optional dataset geo-coding.
    :param xy_names: Optional tuple of the x- and y-coordinate variables in *dataset*. Ignored if *geo_coding* is given.
    :return: The selected variables as a variable name to ``xr.DataArray`` mapping
    """
    geo_coding = geo_coding if geo_coding is not None else GeoCoding.from_dataset(dataset, xy_names=xy_names)
    src_x = geo_coding.x
    x_name, y_name = geo_coding.xy_names
    if var_names is None:
        var_names = [var_name for var_name, var in dataset.data_vars.items()
                     if var_name not in (x_name, y_name) and _is_2d_var(var, src_x)]
    elif isinstance(var_names, str):
        var_names = (var_names,)
    elif len(var_names) == 0:
        raise ValueError(f'empty var_names')
    src_vars = {}
    for var_name in var_names:
        src_var = dataset[var_name]
        if not _is_2d_var(src_var, src_x):
            raise ValueError(
                f"cannot reproject variable {var_name!r} as its shape or dimensions "
                f"do not match those of {x_name!r} and {y_name!r}")
        src_vars[var_name] = src_var
    return src_vars


def select_spatial_subset(dataset: xr.Dataset,
                          bbox: Tuple[int, int, int, int],
                          geo_coding: GeoCoding = None,
                          xy_names: Tuple[str, str] = None) -> Optional[xr.Dataset]:
    """
    Select a spatial subset of *dataset* for the bounding box *bbox*.

    :param dataset: Source dataset.
    :param bbox: Bounding box (i_min, i_min, j_max, j_max) in pixel coordinates.
    :param geo_coding: Optional dataset geo-coding.
    :param xy_names: Optional tuple of the x- and y-coordinate variables in *dataset*. Ignored if *geo_coding* is given.
    :return: Spatial dataset subset
    """
    geo_coding = geo_coding if geo_coding is not None else GeoCoding.from_dataset(dataset, xy_names=xy_names)
    width, height = geo_coding.size
    i_min, j_min, i_max, j_max = bbox
    if i_min > 0 or j_min > 0 or i_max < width - 1 or j_max < height - 1:
        x_dim, y_dim = geo_coding.dims
        i_slice = slice(i_min, i_max + 1)
        j_slice = slice(j_min, j_max + 1)
        return dataset.isel({x_dim: i_slice, y_dim: j_slice})
    return dataset


def _is_2d_var(var: xr.DataArray, two_d_coord_var: xr.DataArray) -> bool:
    return var.ndim >= 2 and var.shape[-2:] == two_d_coord_var.shape and var.dims[-2:] == two_d_coord_var.dims


@nb.jit('float64(float64, float64, float64, float64, float64, float64)',
        nopython=True, nogil=True, inline='always')
def _fdet(px0: float, py0: float, px1: float, py1: float, px2: float, py2: float) -> float:
    return (px0 - px1) * (py0 - py2) - (px0 - px2) * (py0 - py1)


@nb.jit('float64(float64, float64, float64, float64, float64, float64)',
        nopython=True, nogil=True, inline='always')
def _fu(px: float, py: float, px0: float, py0: float, px2: float, py2: float) -> float:
    return (px0 - px) * (py0 - py2) - (py0 - py) * (px0 - px2)


@nb.jit('float64(float64, float64, float64, float64, float64, float64)',
        nopython=True, nogil=True, inline='always')
def _fv(px: float, py: float, px0: float, py0: float, px1: float, py1: float) -> float:
    return (py0 - py) * (px0 - px1) - (px0 - px) * (py0 - py1)


@nb.jit(nopython=True, nogil=True, cache=True)
def _reproject(src_values: np.ndarray,
               src_x: np.ndarray,
               src_y: np.ndarray,
               dst_values: np.ndarray,
               dst_x0: float,
               dst_y0: float,
               dst_res: float,
               uv_delta: float):
    src_width = src_values.shape[-1]
    src_height = src_values.shape[-2]

    dst_width = dst_values.shape[-1]
    dst_height = dst_values.shape[-2]

    dst_px = np.zeros(4, dtype=src_x.dtype)
    dst_py = np.zeros(4, dtype=src_y.dtype)

    u_min = v_min = -uv_delta
    uv_max = 1.0 + 2 * uv_delta

    dst_values[..., :, :] = np.nan

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

            if np.isnan(det_a) or np.isnan(det_b):
                # print('no plane at:', src_i0, src_j0)
                continue

            for dst_j in range(dst_j_min, dst_j_max + 1):
                dst_y = dst_y0 + (dst_j + 0.5) * dst_res
                for dst_i in range(dst_i_min, dst_i_max + 1):
                    dst_x = dst_x0 + (dst_i + 0.5) * dst_res

                    # TODO: use two other combinations,
                    #       if one of the dst_px<n>,dst_py<n> pairs is missing.

                    if not np.isnan(dst_values[..., dst_j, dst_i]):
                        # print('already set:', src_i0, src_j0, '-->', dst_i, dst_j)
                        continue

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


@nb.jit(nopython=True, cache=True)
def compute_source_pixels(src_x: np.ndarray,
                          src_y: np.ndarray,
                          src_i_min: int,
                          src_j_min: int,
                          dst_src_i: np.ndarray,
                          dst_src_j: np.ndarray,
                          dst_x0: float,
                          dst_y0: float,
                          dst_res: float,
                          fractions: bool = False,
                          delta: float = 1e-3):
    src_width = src_x.shape[-1]
    src_height = src_x.shape[-2]

    dst_width = dst_src_i.shape[-1]
    dst_height = dst_src_i.shape[-2]

    dst_px = np.zeros(4, dtype=src_x.dtype)
    dst_py = np.zeros(4, dtype=src_y.dtype)

    dst_src_i[:, :] = np.nan
    dst_src_j[:, :] = np.nan

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

            if np.isnan(det_a) or np.isnan(det_b):
                # print('no plane at:', src_i0, src_j0)
                continue

            for dst_j in range(dst_j_min, dst_j_max + 1):
                dst_y = dst_y0 + (dst_j + 0.5) * dst_res
                for dst_i in range(dst_i_min, dst_i_max + 1):
                    dst_x = dst_x0 + (dst_i + 0.5) * dst_res

                    # TODO: use two other combinations,
                    #       if one of the dst_px<n>,dst_py<n> pairs is missing.

                    src_i = src_j = -1

                    if det_a != 0.0:
                        u = _fu(dst_x, dst_y, dst_p0x, dst_p0y, dst_p2x, dst_p2y) / det_a
                        v = _fv(dst_x, dst_y, dst_p0x, dst_p0y, dst_p1x, dst_p1y) / det_a
                        if u >= u_min and v >= v_min and u + v <= uv_max:
                            if fractions:
                                src_i = src_i0 + u
                                src_j = src_j0 + v
                            else:
                                src_i = src_i0 if u < 0.5 else src_i1
                                src_j = src_j0 if v < 0.5 else src_j1
                    if src_i == -1 and det_b != 0.0:
                        u = _fu(dst_x, dst_y, dst_p3x, dst_p3y, dst_p1x, dst_p1y) / det_b
                        v = _fv(dst_x, dst_y, dst_p3x, dst_p3y, dst_p2x, dst_p2y) / det_b
                        if u >= u_min and v >= v_min and u + v <= uv_max:
                            if fractions:
                                src_i = src_i1 - u
                                src_j = src_j1 - v
                            else:
                                src_i = src_i1 if u < 0.5 else src_i0
                                src_j = src_j1 if v < 0.5 else src_j0
                    if src_i != -1:
                        dst_src_i[dst_j, dst_i] = src_i_min + src_i
                        dst_src_j[dst_j, dst_i] = src_j_min + src_j


@nb.jit(nopython=True, cache=True)
def extract_source_pixels(src_values: np.ndarray,
                          dst_src_i: np.ndarray,
                          dst_src_j: np.ndarray,
                          dst_values: np.ndarray,
                          fill_value: float = np.nan):
    src_width = src_values.shape[-1]
    src_height = src_values.shape[-2]

    dst_width = dst_values.shape[-1]
    dst_height = dst_values.shape[-2]

    # noinspection PyUnusedLocal
    src_i: int = 0
    # noinspection PyUnusedLocal
    src_j: int = 0

    for dst_j in range(dst_height):
        for dst_i in range(dst_width):
            src_i_f = dst_src_i[dst_j, dst_i]
            src_j_f = dst_src_j[dst_j, dst_i]
            if np.isnan(src_i_f) or np.isnan(src_j_f):
                dst_values[..., dst_j, dst_i] = fill_value
            else:
                # TODO: this corresponds to method "nearest": allow for other methods
                src_i = int(src_i_f + 0.49999)
                src_j = int(src_j_f + 0.49999)
                if src_i < 0:
                    src_i = 0
                elif src_i >= src_width:
                    src_i = src_width - 1
                if src_j < 0:
                    src_j = 0
                elif src_j >= src_height:
                    src_j = src_height - 1
                dst_values[..., dst_j, dst_i] = src_values[..., src_j, src_i]


def _millis(seconds: float) -> int:
    return round(1000 * seconds)
