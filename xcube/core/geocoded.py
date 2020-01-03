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
import collections
import math
import time
from typing import Sequence, Tuple, Optional, Union, Mapping

import numba as nb
import numpy as np
import xarray as xr

from xcube.util.dask import compute_array_from_func, ChunkContext

LON_COORD_VAR_NAMES = ('lon', 'long', 'longitude')
LAT_COORD_VAR_NAMES = ('lat', 'latitude')
X_COORD_VAR_NAMES = ('x', 'xc') + LON_COORD_VAR_NAMES
Y_COORD_VAR_NAMES = ('y', 'yc') + LAT_COORD_VAR_NAMES


class GeoCoding(collections.namedtuple('GeoCoding', ['x', 'y', 'x_name', 'y_name', 'is_lon_normalized'])):

    @property
    def size(self) -> Tuple[int, int]:
        height, width = self.x.shape
        return width, height

    @property
    def dims(self) -> Tuple[str, str]:
        y_dim, x_dim = self.x.dims
        return x_dim, y_dim

    @property
    def xy(self) -> Tuple[xr.DataArray, xr.DataArray]:
        return self.x, self.y

    @property
    def xy_names(self) -> Tuple[str, str]:
        return self.x_name, self.y_name

    def derive(self,
               x: xr.DataArray = None,
               y: xr.DataArray = None,
               x_name: str = None,
               y_name: str = None,
               is_lon_normalized: bool = None):
        return GeoCoding(x=x if x is not None else self.x,
                         y=y if y is not None else self.y,
                         x_name=x_name if x_name is not None else self.x_name,
                         y_name=y_name if y_name is not None else self.y_name,
                         is_lon_normalized=is_lon_normalized if is_lon_normalized is not None else self.is_lon_normalized)

    @classmethod
    def from_dataset(cls,
                     dataset: xr.Dataset,
                     xy_names: Tuple[str, str] = None) -> 'GeoCoding':
        """
        Return new geo-coding for given *dataset*.

        :param dataset: Source dataset.
        :param xy_names: Optional tuple of the x- and y-coordinate variables in *dataset*.
        :return: The source dataset's geo-coding.
        """
        x_name, y_name = _get_dataset_xy_names(dataset, xy_names=xy_names)
        x = _get_var(dataset, x_name)
        y = _get_var(dataset, y_name)
        return cls.from_xy((x, y), xy_names=(x_name, y_name))

    @classmethod
    def from_xy(cls,
                xy: Tuple[xr.DataArray, xr.DataArray],
                xy_names: Tuple[str, str] = None) -> 'GeoCoding':
        """
        Return new geo-coding for given *dataset*.

        :param xy: Tuple of x and y coordinate variables.
        :param xy_names: Optional tuple of the x- and y-coordinate variables in *dataset*.
        :return: The source dataset's geo-coding.
        """
        x, y = xy

        if xy_names is None:
            xy_names = x.name, y.name
        x_name, y_name = xy_names
        if x_name is None or y_name is None:
            raise ValueError(f'unable to determine x and y coordinate variable names')

        if x.ndim == 1 and y.ndim == 1:
            x, y = xr.broadcast(y, x)
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError(
                f'coordinate variables {x_name!r} and {y_name!r} must both have either one or two dimensions')

        if x.shape != y.shape or x.dims != y.dims:
            raise ValueError(f"coordinate variables {x_name!r} and {y_name!r} must have same shape and dimensions")

        height, width = x.shape
        if width < 2 or height < 2:
            raise ValueError(f"size in each dimension of {x_name!r} and {y_name!r} must be greater two")

        is_lon_normalized = False
        if x_name in LON_COORD_VAR_NAMES:
            x, is_lon_normalized = _maybe_normalise_2d_lon(x)

        return GeoCoding(x=x, y=y, x_name=x_name, y_name=y_name, is_lon_normalized=is_lon_normalized)

    def pixel_bbox(self,
                   bbox: Tuple[float, float, float, float],
                   border: int = 0,
                   delta: float = 0.0) -> Optional[Tuple[int, int, int, int]]:
        """
        Get a bounding box in pixel coordinates given a bounding box in x,y coordinates.

        :param bbox: Bounding box (x_min, y_min, x_max, y_max) given in the same CS as x and y.
        :param border: Extra border to be added to returned pixel bounding box. Defaults to 0.
        :return: Bounding box in (i_min, j_min, i_max, j_max) in pixel coordinates.
            Returns None if *bbox* isn't intersecting any of the x,y coordinates.
        """
        src_x, src_y = self.xy
        dst_x_min, dst_y_min, dst_x_max, dst_y_max = bbox
        if self.is_lon_normalized:
            if dst_x_min < 0.0:
                dst_x_min += 360.0
            if dst_x_max < 0.0:
                dst_x_max += 360.0
        if dst_x_min > dst_x_max:
            dst_x_max += 360.0
        src_bbox = np.logical_and(np.logical_and(src_x >= dst_x_min - delta, src_x <= dst_x_max + delta),
                                  np.logical_and(src_y >= dst_y_min - delta, src_y <= dst_y_max + delta))
        dim_y, dim_x = src_x.dims
        src_i = src_x[dim_x].where(src_bbox)
        src_j = src_y[dim_y].where(src_bbox)
        src_i_min = src_i.min()
        src_i_max = src_i.max()
        src_j_min = src_j.min()
        src_j_max = src_j.max()
        if not np.isfinite(src_i_min) or not np.isfinite(src_j_min) \
                or not np.isfinite(src_i_max) or not np.isfinite(src_j_max):
            return None
        height, width = src_x.shape
        src_i1 = int(src_i_min - border)
        src_i2 = int(src_i_max + border)
        src_j1 = int(src_j_min - border)
        src_j2 = int(src_j_max + border)
        if src_i1 < 0:
            src_i1 = 0
        if src_j1 < 0:
            src_j1 = 0
        if src_i2 >= width:
            src_i2 = width - 1
        if src_j2 >= height:
            src_j2 = height - 1
        return src_i1, src_j1, src_i2, src_j2


class ImageGeom(collections.namedtuple('GeoCoding', ['width', 'height', 'x_min', 'y_min', 'res'])):
    @property
    def is_crossing_antimeridian(self):
        return self.x_min > self.x_max

    @property
    def x_max(self):
        x_max = self.x_min + self.res * self.width
        if x_max > 180.0:
            x_max -= 360.0
        return x_max

    @property
    def y_max(self):
        return self.y_min + self.res * self.height

    @property
    def bbox(self):
        return self.x_min, self.y_min, self.x_max, self.y_max

    @property
    def size(self) -> Tuple[int, int]:
        return self.width, self.height


def reproject_dataset(src_ds: xr.Dataset,
                      var_names: Union[str, Sequence[str]] = None,
                      geo_coding: GeoCoding = None,
                      xy_names: Tuple[str, str] = None,
                      tile_size: Union[int, Tuple[int, int]] = None,
                      output_geom: ImageGeom = None,
                      delta: float = 1e-3) -> Optional[xr.Dataset]:
    """
    Reproject *dataset* using its per-pixel x,y coordinates or the given *geo_coding*.

    :param src_ds: Source dataset.
    :param var_names: Optional variable name or sequence of variable names.
    :param geo_coding: Optional dataset geo-coding.
    :param xy_names: Optional tuple of the x- and y-coordinate variables in *dataset*. Ignored if *geo_coding* is given.
    :param tile_size: Optional tile size, an integer or tuple (tile_width, tile_height).
    :param output_geom: Optional output geometry. If not given, output geometry will be computed
        to spatially fit *dataset* and to retain its spatial resolution.
    :param delta:
    :return:
    """
    src_geo_coding = geo_coding if geo_coding is not None else GeoCoding.from_dataset(src_ds, xy_names=xy_names)
    src_x, src_y = src_geo_coding.xy
    src_x_dim, src_y_dim = src_geo_coding.dims
    src_x_name, src_y_name = src_geo_coding.xy_names
    is_lon_normalized = src_geo_coding.is_lon_normalized

    if isinstance(tile_size, int):
        tile_size = tile_size, tile_size

    src_i_min_0 = 0
    src_j_min_0 = 0
    if output_geom is None:
        output_geom = compute_output_geom(src_ds, geo_coding=src_geo_coding)
    else:
        src_bbox = src_geo_coding.pixel_bbox(output_geom.bbox, border=1, delta=output_geom.res)
        if src_bbox is None:
            return None
        src_i_min_0, src_j_min_0 = src_bbox[0:2]
        if is_lon_normalized:
            src_ds = src_ds.copy()
            src_ds[src_x_name] = src_x
        src_ds = select_spatial_subset(src_ds, src_bbox, geo_coding=src_geo_coding)
        src_x, src_y = src_ds[src_x_name], src_ds[src_y_name]
        src_geo_coding = src_geo_coding.derive(x=src_x, y=src_y)

    src_vars = select_variables(src_ds, var_names, geo_coding=src_geo_coding)

    dst_width, dst_height, dst_x_min, dst_y_min, dst_res = output_geom

    dst_x_var = xr.DataArray(np.linspace(dst_x_min, dst_x_min + dst_res * (dst_width - 1),
                                         num=dst_width, dtype=np.float64), dims=src_x_name)
    dst_y_var = xr.DataArray(np.linspace(dst_y_min, dst_y_min + dst_res * (dst_height - 1),
                                         num=dst_height, dtype=np.float64), dims=src_y_name)
    dst_coords = {
        src_x_name: _denormalize_lon(dst_x_var) if is_lon_normalized else dst_x_var,
        src_y_name: dst_y_var
    }
    dst_dims = (src_y_name, src_x_name)

    if tile_size is None:
        src_x_values = src_x.values
        src_y_values = src_y.values
    else:
        src_x_values = src_x
        src_y_values = src_y

    dst_vars = dict()
    for src_var_name, src_var in src_vars.items():
        dst_var_shape = src_var.shape[0:-2] + (dst_height, dst_width)
        dst_var_dims = src_var.dims[0:-2] + dst_dims
        dst_var_coords = dict(src_var.coords)
        dst_var_coords.update(**dst_coords)
        if tile_size is None:
            dst_var_array = np.full(dst_var_shape, np.nan, dtype=src_var.dtype)
            reproject(src_var.values,
                      src_x_values,
                      src_y_values,
                      src_i_min_0,
                      src_j_min_0,
                      dst_var_array,
                      dst_x_min,
                      dst_y_min,
                      dst_res,
                      delta=delta)
        else:
            def reproject_func(context: ChunkContext) -> np.ndarray:
                try:
                    dst_block = np.full(context.chunk_shape, np.nan, dtype=context.dtype)
                    dst_y_slice, dst_x_slice = context.chunk_slices
                    dst_chunk_bbox = (dst_x_min + dst_x_slice.start * dst_res,
                                      dst_y_min + dst_y_slice.start * dst_res,
                                      dst_x_min + dst_x_slice.stop * dst_res,
                                      dst_y_min + dst_y_slice.stop * dst_res)
                    t1 = time.perf_counter()
                    src_chunk_bbox = src_geo_coding.pixel_bbox(dst_chunk_bbox, border=1, delta=dst_res)
                    t2 = time.perf_counter()
                    if src_chunk_bbox is None:
                        return dst_block
                    src_i_min, src_j_min, src_i_max, src_j_max = src_chunk_bbox
                    src_i_slice = slice(src_i_min, src_i_max + 1)
                    src_j_slice = slice(src_j_min, src_j_max + 1)
                    src_indexers = {src_x_dim: src_i_slice, src_y_dim: src_j_slice}
                    src_values = src_var.isel(**src_indexers).values
                    src_x_values = src_x.isel(**src_indexers).values
                    src_y_values = src_y.isel(**src_indexers).values
                    t3 = time.perf_counter()
                    reproject(src_values,
                              src_x_values,
                              src_y_values,
                              src_i_min_0 + src_i_min,
                              src_j_min_0 + src_j_min,
                              dst_block,
                              dst_x_min + dst_x_slice.start * dst_res,
                              dst_y_min + dst_y_slice.start * dst_res,
                              dst_res,
                              delta=delta)
                    t4 = time.perf_counter()
                    print(f'chunk {context.name}-{context.chunk_index}, shape {context.chunk_shape} '
                          f'took {t2 - t1}, {t3 - t2}, {t4 - t3} seconds, total {t4 - t1}')
                    return dst_block
                except BaseException as e:
                    print(80 * '#')
                    print(e)
                    print(80 * '#')
                    raise e

            tile_height, tile_width = tile_size
            dst_var_chunks = src_var.shape[0:-2] + (tile_height, tile_width)
            dst_var_array = compute_array_from_func(reproject_func,
                                                    dst_var_shape,
                                                    dst_var_chunks,
                                                    src_var.dtype,
                                                    name=src_var_name)

        dst_var = xr.DataArray(dst_var_array,
                               dims=dst_var_dims,
                               coords=dst_var_coords,
                               attrs=src_var.attrs)
        dst_vars[src_var_name] = dst_var
    return xr.Dataset(dst_vars, attrs=src_ds.attrs)


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
        src_i_slice = slice(i_min, i_max + 1)
        src_j_slice = slice(j_min, j_max + 1)
        dim_x, dim_y = geo_coding.dims
        indexers = {dim_y: src_j_slice, dim_x: src_i_slice}
        return xr.Dataset({var_name: var.isel(**indexers) for var_name, var in dataset.variables.items()
                           if var.shape == (height, width) and var.dims[-2:] == (dim_y, dim_x)})
    return dataset


def compute_output_geom(dataset: xr.Dataset,
                        geo_coding: GeoCoding = None,
                        xy_names: Tuple[str, str] = None,
                        oversampling: float = 1.0,
                        denom_x: int = 1,
                        denom_y: int = 1,
                        delta: float = 1e-10) -> ImageGeom:
    """
    Compute image geometry for a rectified output image that retains the source's bounding box and its
    spatial resolution in both x and y.

    :param dataset: Source dataset.
    :param geo_coding: Optional dataset geo-coding.
    :param xy_names: Optional tuple of the x- and y-coordinate variables in *dataset*. Ignored if *geo_coding* is given.
    :param oversampling:
    :param denom_x:
    :param denom_y:
    :param delta:
    :return: A new image geometry (class ImageGeometry).
    """
    geo_coding = geo_coding if geo_coding is not None else GeoCoding.from_dataset(dataset, xy_names=xy_names)
    src_x, src_y = geo_coding.xy
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


def _get_dataset_xy_names(dataset: xr.Dataset, xy_names: Tuple[str, str] = None) -> Tuple[str, str]:
    x_name, y_name = xy_names if xy_names is not None else (None, None)
    return (_get_coord_var_name(dataset, x_name, X_COORD_VAR_NAMES, 'x'),
            _get_coord_var_name(dataset, y_name, Y_COORD_VAR_NAMES, 'y'))


def _get_coord_var_name(dataset: xr.Dataset, coord_name: Optional[str], coord_var_names: Sequence[str], dim_name: str):
    if not coord_name:
        coord_name = _find_coord_var_name(dataset, coord_var_names, 2)
        if coord_name is None:
            coord_name = _find_coord_var_name(dataset, coord_var_names, 1)
            if not coord_name:
                raise ValueError(f'cannot detect {dim_name!r}-coordinate variable in dataset')
    elif coord_name not in dataset:
        raise ValueError(f'missing coordinate variable {coord_name!r} in dataset')
    return coord_name


def _find_coord_var_name(dataset: xr.Dataset, coord_var_names: Sequence[str], ndim: int) -> Optional[str]:
    for coord_var_name in coord_var_names:
        if coord_var_name in dataset and dataset[coord_var_name].ndim == ndim:
            return coord_var_name
    return None


def _get_var(src_ds: xr.Dataset, name: str) -> xr.DataArray:
    if name not in src_ds:
        raise ValueError(f'missing 2D coordinate variable {name!r}')
    return src_ds[name]


def _is_2d_var(var: xr.DataArray, two_d_coord_var: xr.DataArray) -> bool:
    return var.ndim >= 2 and var.shape[-2:] == two_d_coord_var.shape and var.dims[-2:] == two_d_coord_var.dims


def _is_crossing_antimeridian(lon_var: xr.DataArray):
    dim_y, dim_x = lon_var.dims
    # noinspection PyTypeChecker
    return abs(lon_var.diff(dim=dim_x)).max() > 180.0 or \
           abs(lon_var.diff(dim=dim_y)).max() > 180.0


def _maybe_normalise_2d_lon(lon_var: xr.DataArray):
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
              src_i_min: int,
              src_j_min: int,
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
                dst_y = dst_y0 + dst_j * dst_res
                for dst_i in range(dst_i_min, dst_i_max + 1):
                    dst_x = dst_x0 + dst_i * dst_res

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
            for dst_j in range(dst_j_min, dst_j_max + 1):
                dst_y = dst_y0 + dst_j * dst_res
                for dst_i in range(dst_i_min, dst_i_max + 1):
                    dst_x = dst_x0 + dst_i * dst_res

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
