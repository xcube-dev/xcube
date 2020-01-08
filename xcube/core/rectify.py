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

import math
import time
from typing import Sequence, Tuple, Optional, Union, Mapping

import numba as nb
import numpy as np
import xarray as xr

from xcube.util.dask import compute_array_from_func, ChunkContext, get_chunk_sizes, get_chunk_iterators

LON_COORD_VAR_NAMES = ('lon', 'long', 'longitude')
LAT_COORD_VAR_NAMES = ('lat', 'latitude')
X_COORD_VAR_NAMES = ('x', 'xc') + LON_COORD_VAR_NAMES
Y_COORD_VAR_NAMES = ('y', 'yc') + LAT_COORD_VAR_NAMES


class GeoCoding:

    def __init__(self,
                 x: xr.DataArray,
                 y: xr.DataArray,
                 x_name: str,
                 y_name: str,
                 is_lon_normalized: bool = False):
        self._x = x
        self._y = y
        self._x_name = x_name
        self._y_name = y_name
        self._is_lon_normalized = is_lon_normalized

    @property
    def x(self) -> xr.DataArray:
        return self._x

    @property
    def y(self) -> xr.DataArray:
        return self._y

    @property
    def xy(self) -> Tuple[xr.DataArray, xr.DataArray]:
        return self.x, self.y

    @property
    def x_name(self) -> str:
        return self._x_name

    @property
    def y_name(self) -> str:
        return self._y_name

    @property
    def xy_names(self) -> Tuple[str, str]:
        return self.x_name, self.y_name

    @property
    def is_lon_normalized(self) -> bool:
        return self._is_lon_normalized

    @property
    def size(self) -> Tuple[int, int]:
        height, width = self.x.shape
        return width, height

    @property
    def dims(self) -> Tuple[str, str]:
        y_dim, x_dim = self.x.dims
        return str(x_dim), str(y_dim)

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
                         is_lon_normalized=is_lon_normalized if is_lon_normalized is not None
                         else self.is_lon_normalized)

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
            y, x = xr.broadcast(y, x)
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

    def ij_bbox(self,
                xy_bbox: Tuple[float, float, float, float],
                xy_border: float = 0.0,
                ij_border: int = 0,
                gu: bool = False) -> Tuple[int, int, int, int]:
        """
        Compute bounding box in i,j pixel coordinates given a bounding box *xy_bbox* in x,y coordinates.

        :param xy_bbox: Bounding box (x_min, y_min, x_max, y_max) given in the same CS as x and y.
        :param xy_border: If non-zero, grows the bounding box *xy_bbox* before using it for comparisons. Defaults to 0.
        :param ij_border: If non-zero, grows the returned i,j bounding box and clips it to size. Defaults to 0.
        :param gu: Use generic ufunc for the computation (may be faster). Defaults to False.
        :return: Bounding box in (i_min, j_min, i_max, j_max) in pixel coordinates.
            Returns None if *xy_bbox* isn't intersecting any of the x,y coordinates.
        """
        xy_bboxes = np.array([xy_bbox], dtype=np.float64)
        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        self.ij_bboxes(xy_bboxes, xy_border=xy_border, ij_border=ij_border, ij_bboxes=ij_bboxes, gu=gu)
        # noinspection PyTypeChecker
        return tuple(map(int, ij_bboxes[0]))

    def ij_bboxes(self,
                  xy_bboxes: np.ndarray,
                  xy_border: float = 0.0,
                  ij_bboxes: np.ndarray = None,
                  ij_border: int = 0,
                  gu: bool = False) -> np.ndarray:
        """
        Compute bounding boxes in i,j pixel coordinates given bounding boxes *xy_bboxes* in x,y coordinates.

        :param xy_bboxes: Numpy array of x,y bounding boxes [[x_min, y_min, x_max, y_max], ...]
            given in the same CS as x and y.
        :param xy_border: If non-zero, grows the bounding box *xy_bbox* before using it for comparisons. Defaults to 0.
        :param ij_bboxes: Numpy array of pixel i,j bounding boxes [[x_min, y_min, x_max, y_max], ...].
            If given, must have same shape as *xy_bboxes*.
        :param ij_border: If non-zero, grows the returned i,j bounding box and clips it to size. Defaults to 0.
        :param gu: Use generic ufunc for the computation (may be faster). Defaults to False.
        :return: Bounding box in (i_min, j_min, i_max, j_max) in pixel coordinates.
            Returns None if *xy_bbox* isn't intersecting any of the x,y coordinates.
        """
        if self.is_lon_normalized:
            xy_bboxes = xy_bboxes.copy()
            c0 = xy_bboxes[:, 0]
            c2 = xy_bboxes[:, 2]
            c0 = np.where(c0 < 0.0, c0 + 360.0, c0)
            c2 = np.where(c2 < 0.0, c2 + 360.0, c2)
            xy_bboxes[:, 0] = c0
            xy_bboxes[:, 2] = c2

        c0 = xy_bboxes[:, 0]
        c2 = xy_bboxes[:, 2]
        cond = c0 > c2
        if np.any(cond):
            xy_bboxes = xy_bboxes.copy()
            xy_bboxes[:, 2] += 360.0

        if ij_bboxes is None:
            ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        else:
            ij_bboxes[:, :] = -1
        if gu:
            gu_compute_ij_bboxes(self.x.values,
                                 self.y.values,
                                 xy_bboxes,
                                 xy_border,
                                 ij_border,
                                 ij_bboxes)
        else:
            compute_ij_bboxes(self.x.values,
                              self.y.values,
                              xy_bboxes,
                              xy_border,
                              ij_border,
                              ij_bboxes)
        return ij_bboxes


class ImageGeom:

    @classmethod
    def from_dataset(cls,
                     dataset: xr.Dataset,
                     geo_coding: GeoCoding = None,
                     xy_names: Tuple[str, str] = None,
                     xy_oversampling: float = 1.0,
                     xy_eps: float = 1e-10,
                     ij_denom: Union[int, Tuple[int, int]] = None):
        """
        Compute image geometry for a rectified output image that retains the source's x,y bounding box and its
        spatial resolution.

        The spatial resolution is computed as the minimum (greater than *xy_eps*) of the absolute value of
        x,y-deltas in i- and j-direction with i denoting columns, and j the image's rows.

        :param dataset: Source dataset.
        :param geo_coding: Optional dataset geo-coding.
        :param xy_names: Optional tuple of the x- and y-coordinate variables in *dataset*.
            Ignored if *geo_coding* is given.
        :param xy_oversampling: The computed resolution is divided by this value while the computed size is multiplied.
        :param xy_eps: Computed resolutions must be greater than this value, to avoid computing a zero resolution.
        :param ij_denom: if given, and greater one, width and height will be multiples of ij_denom
        :return: A new image geometry.
        """
        return _compute_output_geom(dataset,
                                    geo_coding=geo_coding,
                                    xy_names=xy_names,
                                    xy_oversampling=xy_oversampling,
                                    xy_eps=xy_eps,
                                    ij_denom=ij_denom)

    def __init__(self,
                 size: Union[int, Tuple[int, int]],
                 tile_size: Union[int, Tuple[int, int]] = None,
                 x_min: float = 0.0,
                 y_min: float = 0.0,
                 xy_res: float = 1.0):

        if isinstance(size, int):
            w, h = size, size
        else:
            w, h = size

        if isinstance(tile_size, int):
            tw, th = tile_size, tile_size
        elif tile_size is not None:
            tw, th = tile_size
        else:
            tw, th = w, h

        self._size = w, h
        self._tile_size = min(w, tw) or w, min(h, th) or h
        self._x_min = x_min
        self._y_min = y_min
        self._xy_res = xy_res

    def derive(self,
               size: Tuple[int, int] = None,
               tile_size: Union[int, Tuple[int, int]] = None,
               x_min: float = None,
               y_min: float = None,
               xy_res: float = None):
        return ImageGeom(self.size if size is None else size,
                         tile_size=self.tile_size if tile_size is None else tile_size,
                         x_min=self.x_min if x_min is None else x_min,
                         y_min=self.y_min if y_min is None else y_min,
                         xy_res=self.xy_res if xy_res is None else xy_res)

    @property
    def size(self) -> Tuple[int, int]:
        return self._size

    @property
    def width(self) -> int:
        return self._size[0]

    @property
    def height(self) -> int:
        return self._size[1]

    @property
    def is_tiled(self) -> bool:
        return self._size != self._tile_size

    @property
    def tile_size(self) -> Tuple[int, int]:
        return self._tile_size

    @property
    def tile_width(self) -> int:
        return self._tile_size[0]

    @property
    def tile_height(self) -> int:
        return self._tile_size[1]

    @property
    def is_crossing_antimeridian(self) -> bool:
        # TODO (forman): this test is only valid for a geographical CRS
        return self.x_min > self.x_max

    @property
    def x_min(self) -> float:
        return self._x_min

    @property
    def x_max(self) -> float:
        x_max = self.x_min + self.xy_res * self.width
        if x_max > 180.0:
            # TODO (forman): this test is only valid for a geographical CRS
            x_max -= 360.0
        return x_max

    @property
    def y_min(self) -> float:
        return self._y_min

    @property
    def y_max(self) -> float:
        return self.y_min + self.xy_res * self.height

    @property
    def xy_res(self) -> float:
        return self._xy_res

    @property
    def xy_bbox(self) -> Tuple[float, float, float, float]:
        return self.x_min, self.y_min, self.x_max, self.y_max

    @property
    def xy_bboxes(self) -> np.ndarray:
        xy_offset = np.array([self.x_min, self.y_min, self.x_min, self.y_min])
        tile_bboxes = self.ij_bboxes
        return xy_offset + self.xy_res * tile_bboxes

    @property
    def ij_bboxes(self) -> np.ndarray:
        chunk_sizes = get_chunk_sizes((self.height, self.width),
                                      (self.tile_height, self.tile_width))
        _, _, chunk_slice_tuples = get_chunk_iterators(chunk_sizes)
        chunk_slice_tuples = tuple(chunk_slice_tuples)
        n = len(chunk_slice_tuples)
        tile_bboxes = np.ndarray((n, 4), dtype=np.int64)
        for i in range(n):
            y_slice, x_slice = chunk_slice_tuples[i]
            tile_bboxes[i, 0] = x_slice.start
            tile_bboxes[i, 1] = y_slice.start
            tile_bboxes[i, 2] = x_slice.stop - 1
            tile_bboxes[i, 3] = y_slice.stop - 1
        return tile_bboxes


def rectify_dataset(dataset: xr.Dataset,
                    var_names: Union[str, Sequence[str]] = None,
                    geo_coding: GeoCoding = None,
                    xy_names: Tuple[str, str] = None,
                    output_geom: ImageGeom = None,
                    is_y_axis_inverted: bool = False,
                    tile_size: Union[int, Tuple[int, int]] = None,
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
    :param uv_delta: A normalized value that is used to determine whether x,y coordinates in the output are contained
        in the triangles defined by the input x,y coordinates.
    :return: a reprojected dataset, or None if the requested output does not intersect with *dataset*.
    """
    src_geo_coding = geo_coding if geo_coding is not None else GeoCoding.from_dataset(dataset, xy_names=xy_names)
    src_x, src_y = src_geo_coding.xy
    src_x_name, src_y_name = src_geo_coding.xy_names
    is_lon_normalized = src_geo_coding.is_lon_normalized

    src_i_min_0 = 0
    src_j_min_0 = 0
    if output_geom is None:
        output_geom = ImageGeom.from_dataset(dataset, geo_coding=src_geo_coding)
    else:
        src_bbox = src_geo_coding.ij_bbox(output_geom.xy_bbox, ij_border=1, xy_border=output_geom.xy_res)
        if src_bbox[0] == -1:
            return None
        src_i_min_0, src_j_min_0 = src_bbox[0:2]
        if is_lon_normalized:
            dataset = dataset.copy()
            dataset[src_x_name] = src_x
        dataset = select_spatial_subset(dataset, src_bbox, geo_coding=src_geo_coding)
        src_x, src_y = dataset[src_x_name], dataset[src_y_name]
        src_geo_coding = src_geo_coding.derive(x=src_x, y=src_y)

    if tile_size is not None:
        output_geom = output_geom.derive(tile_size=tile_size)

    src_vars = select_variables(dataset, var_names, geo_coding=src_geo_coding)

    dst_width, dst_height = output_geom.size
    dst_x_min, dst_y_min = output_geom.x_min, output_geom.y_min
    dst_xy_res = output_geom.xy_res

    dst_x_var_values = np.linspace(dst_x_min, dst_x_min + dst_xy_res * (dst_width - 1),
                                   num=dst_width, dtype=np.float64)
    dst_y_var_values = np.linspace(dst_y_min, dst_y_min + dst_xy_res * (dst_height - 1),
                                   num=dst_height, dtype=np.float64)
    dst_x_var = xr.DataArray(dst_x_var_values, dims=src_x_name)
    dst_y_var = xr.DataArray(dst_y_var_values, dims=src_y_name)
    if is_lon_normalized:
        dst_x_var = _denormalize_lon(dst_x_var)
    if is_y_axis_inverted:
        dst_y_var = dst_y_var[::-1]
    dst_coords = {src_x_name: dst_x_var, src_y_name: dst_y_var}
    dst_dims = (src_y_name, src_x_name)

    is_output_tiled = output_geom.is_tiled
    if not is_output_tiled:
        src_x.load()
        src_y.load()

        def _get_dst_var_array(src_var):
            dst_var_array = np.full(dst_var_shape, np.nan, dtype=src_var.dtype)
            _reproject(src_var.values,
                       src_x.values,
                       src_y.values,
                       src_i_min_0,
                       src_j_min_0,
                       dst_var_array,
                       dst_x_min,
                       dst_y_min,
                       dst_xy_res,
                       uv_delta=uv_delta)
            if is_y_axis_inverted:
                dst_var_array = dst_var_array[..., ::-1, :]
            return dst_var_array
    else:
        # This is NOT faster:
        # src_x.load()
        # src_y.load()

        dst_xy_bboxes = output_geom.xy_bboxes
        src_ij_bboxes = src_geo_coding.ij_bboxes(dst_xy_bboxes, xy_border=dst_xy_res, ij_border=1)

        def _get_dst_var_array(src_var):

            # This is NOT faster:
            src_var = src_var.compute()

            def _reproject_func(context: ChunkContext) -> np.ndarray:
                try:
                    dst_block = np.full(dst_var_shape[:-2] + context.chunk_shape, np.nan, dtype=context.dtype)
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
                    t1 = time.perf_counter()
                    src_var_values = src_var[..., src_j_slice, src_i_slice].values
                    src_x_values = src_x[src_j_slice, src_i_slice].values
                    src_y_values = src_y[src_j_slice, src_i_slice].values
                    t2 = time.perf_counter()
                    _reproject(src_var_values,
                               src_x_values,
                               src_y_values,
                               src_i_min_0 + src_i_min,
                               src_j_min_0 + src_j_min,
                               dst_block,
                               dst_x_min + dst_x_slice.start * dst_xy_res,
                               dst_y_min + dst_y_slice.start * dst_xy_res,
                               dst_xy_res,
                               uv_delta=uv_delta)
                    t3 = time.perf_counter()
                    print(f'target chunk {context.name}-{context.chunk_index}, shape {context.chunk_shape} '
                          f'for source shape {src_i_max - src_i_min + 1, src_j_max - src_j_min + 1} '
                          f'took {_millis(t2 - t1)}, {_millis(t3 - t2)} milliseconds, total {_millis(t3 - t1)}')
                    if is_y_axis_inverted:
                        dst_block = dst_block[..., ::-1, :]
                    return dst_block
                except BaseException as e:
                    print(80 * '#')
                    print(e)
                    print(80 * '#')
                    raise e

            tile_width, tile_height = output_geom.tile_size
            dst_var_chunks = src_var.shape[0:-2] + (tile_height, tile_width)
            return compute_array_from_func(_reproject_func,
                                           dst_var_shape,
                                           dst_var_chunks,
                                           src_var.dtype,
                                           name=src_var_name)

    dst_vars = dict()
    for src_var_name, src_var in src_vars.items():
        dst_var_shape = src_var.shape[0:-2] + (dst_height, dst_width)
        dst_var_dims = src_var.dims[0:-2] + dst_dims
        dst_var_coords = dict(src_var.coords)
        dst_var_coords.update(**dst_coords)
        dst_var_array = _get_dst_var_array(src_var)
        dst_var = xr.DataArray(dst_var_array,
                               dims=dst_var_dims,
                               coords=dst_var_coords,
                               attrs=src_var.attrs)
        dst_vars[src_var_name] = dst_var
    return xr.Dataset(dst_vars, attrs=dataset.attrs)


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


def _compute_output_geom(dataset: xr.Dataset,
                         geo_coding: GeoCoding = None,
                         xy_names: Tuple[str, str] = None,
                         xy_oversampling: float = 1.0,
                         xy_eps: float = 1e-10,
                         ij_denom: Union[int, Tuple[int, int]] = None) -> ImageGeom:
    i_denom, j_denom = ((ij_denom, ij_denom) if isinstance(ij_denom, int) else ij_denom) or (1, 1)
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
    src_x_res = float(src_x_diff.where(src_x_diff > xy_eps).min())
    src_y_res = float(src_y_diff.where(src_y_diff > xy_eps).min())
    src_xy_res = min(src_x_res, src_y_res) / (math.sqrt(2.0) * xy_oversampling)
    src_x_min = float(src_x.min())
    src_x_max = float(src_x.max())
    src_y_min = float(src_y.min())
    src_y_max = float(src_y.max())
    dst_width = 1 + math.floor((src_x_max - src_x_min) / src_xy_res)
    dst_height = 1 + math.floor((src_y_max - src_y_min) / src_xy_res)
    dst_width = i_denom * ((dst_width + i_denom - 1) // i_denom)
    dst_height = j_denom * ((dst_height + j_denom - 1) // j_denom)
    return ImageGeom((dst_width, dst_height),
                     x_min=src_x_min,
                     y_min=src_y_min,
                     xy_res=src_xy_res)


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
def _reproject(src_values: np.ndarray,
               src_x: np.ndarray,
               src_y: np.ndarray,
               src_i_min: int,
               src_j_min: int,
               dst_values: np.ndarray,
               dst_x0: float,
               dst_y0: float,
               dst_res: float,
               uv_delta: float = 1e-3):
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


@nb.jit(nopython=True, cache=True, target='cpu')
def compute_ij_bboxes(x_image: np.ndarray,
                      y_image: np.ndarray,
                      xy_bboxes: np.ndarray,
                      xy_border: float,
                      ij_border: int,
                      ij_bboxes: np.ndarray):
    h = x_image.shape[0]
    w = x_image.shape[1]
    n = xy_bboxes.shape[0]
    for k in range(n):
        ij_bbox = ij_bboxes[k]
        xy_bbox = xy_bboxes[k]
        x_min = xy_bbox[0] - xy_border
        y_min = xy_bbox[1] - xy_border
        x_max = xy_bbox[2] + xy_border
        y_max = xy_bbox[3] + xy_border
        for j in range(h):
            for i in range(w):
                x = x_image[j, i]
                if x_min <= x <= x_max:
                    y = y_image[j, i]
                    if y_min <= y <= y_max:
                        i_min = ij_bbox[0]
                        j_min = ij_bbox[1]
                        i_max = ij_bbox[2]
                        j_max = ij_bbox[3]
                        ij_bbox[0] = i if i_min < 0 else min(i_min, i)
                        ij_bbox[1] = j if j_min < 0 else min(j_min, j)
                        ij_bbox[2] = i if i_max < 0 else max(i_max, i)
                        ij_bbox[3] = j if j_max < 0 else max(j_max, j)
        if ij_border != 0 and ij_bbox[0] != -1:
            i_min = ij_bbox[0] - ij_border
            j_min = ij_bbox[1] - ij_border
            i_max = ij_bbox[2] + ij_border
            j_max = ij_bbox[3] + ij_border
            if i_min < 0:
                i_min = 0
            if j_min < 0:
                j_min = 0
            if i_max >= w:
                i_max = w - 1
            if j_max >= h:
                j_max = h - 1
            ij_bbox[0] = i_min
            ij_bbox[1] = j_min
            ij_bbox[2] = i_max
            ij_bbox[3] = j_max


# This is NOT faster:
@nb.guvectorize([(nb.float64[:, :],
                  nb.float64[:, :],
                  nb.float64[:, :],
                  nb.float64,
                  nb.int64,
                  nb.int64[:, :])],
                '(h,w),(h,w),(n,m),(),()->(n,m)',
                cache=True, target='cpu')
def gu_compute_ij_bboxes(x_image: np.ndarray,
                         y_image: np.ndarray,
                         xy_bboxes: np.ndarray,
                         xy_border: float,
                         ij_border: int,
                         ij_bboxes: np.ndarray):
    h = x_image.shape[0]
    w = x_image.shape[1]
    n = xy_bboxes.shape[0]
    for k in range(n):
        x_min = xy_bboxes[k, 0] - xy_border
        y_min = xy_bboxes[k, 1] - xy_border
        x_max = xy_bboxes[k, 2] + xy_border
        y_max = xy_bboxes[k, 3] + xy_border
        for j in range(h):
            for i in range(w):
                x = x_image[j, i]
                if x_min <= x <= x_max:
                    y = y_image[j, i]
                    if y_min <= y <= y_max:
                        i_min = ij_bboxes[k, 0]
                        j_min = ij_bboxes[k, 1]
                        i_max = ij_bboxes[k, 2]
                        j_max = ij_bboxes[k, 3]
                        ij_bboxes[k, 0] = i if i_min < 0 else min(i_min, i)
                        ij_bboxes[k, 1] = j if j_min < 0 else min(j_min, j)
                        ij_bboxes[k, 2] = i if i_max < 0 else max(i_max, i)
                        ij_bboxes[k, 3] = j if j_max < 0 else max(j_max, j)
        if ij_border != 0 and ij_bboxes[k, 0] != -1:
            i_min = ij_bboxes[k, 0] - ij_border
            j_min = ij_bboxes[k, 1] - ij_border
            i_max = ij_bboxes[k, 2] + ij_border
            j_max = ij_bboxes[k, 3] + ij_border
            if i_min < 0:
                i_min = 0
            if j_min < 0:
                j_min = 0
            if i_max >= w:
                i_max = w - 1
            if j_max >= h:
                j_max = h - 1
            ij_bboxes[k, 0] = i_min
            ij_bboxes[k, 1] = j_min
            ij_bboxes[k, 2] = i_max
            ij_bboxes[k, 3] = j_max


def _millis(seconds: float) -> int:
    return round(1000 * seconds)
