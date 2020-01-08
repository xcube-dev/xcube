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
from typing import Tuple, Union

import numpy as np
import xarray as xr

from xcube.core.geocoding import GeoCoding
from xcube.util.dask import get_chunk_sizes, get_chunk_iterators

LON_COORD_VAR_NAMES = ('lon', 'long', 'longitude')
LAT_COORD_VAR_NAMES = ('lat', 'latitude')
X_COORD_VAR_NAMES = ('x', 'xc') + LON_COORD_VAR_NAMES
Y_COORD_VAR_NAMES = ('y', 'yc') + LAT_COORD_VAR_NAMES


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

        # TODO (forman): validate args & kwargs

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
