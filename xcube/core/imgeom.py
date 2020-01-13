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
from typing import Tuple, Union, Mapping

import numpy as np
import xarray as xr

from xcube.core.geocoding import GeoCoding, denormalize_lon
from xcube.util.dask import get_chunk_sizes, get_block_iterators

_LON_ATTRS = dict(long_name='longitude', standard_name='longitude', units='degrees_east')
_LAT_ATTRS = dict(long_name='latitude', standard_name='latitude', units='degrees_north')


class ImageGeom:

    def __init__(self,
                 size: Union[int, Tuple[int, int]],
                 tile_size: Union[int, Tuple[int, int]] = None,
                 x_min: float = 0.0,
                 y_min: float = 0.0,
                 xy_res: float = 1.0,
                 is_geo_crs: bool = False):

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

        if w <= 0 or h <= 0:
            raise ValueError('invalid size')

        if tw <= 0 or th <= 0:
            raise ValueError('invalid tile_size')

        if xy_res <= 0:
            raise ValueError('invalid xy_res')

        if is_geo_crs:
            if w * xy_res > 360.0:
                raise ValueError('invalid size, xy_res combination')
            if x_min < -180.0 or x_min > 180.0:
                raise ValueError('invalid x_min')
            if y_min < -90.0 or y_min > 90.0 or y_min + h * xy_res > 90.0:
                raise ValueError('invalid y_min')

        self._size = w, h
        self._tile_size = min(w, tw) or w, min(h, th) or h
        self._x_min = x_min
        self._y_min = y_min
        self._xy_res = xy_res
        self._is_geo_crs = is_geo_crs

    def derive(self,
               size: Tuple[int, int] = None,
               tile_size: Union[int, Tuple[int, int]] = None,
               x_min: float = None,
               y_min: float = None,
               xy_res: float = None,
               is_geo_crs: bool = None):
        return ImageGeom(self.size if size is None else size,
                         tile_size=self.tile_size if tile_size is None else tile_size,
                         x_min=self.x_min if x_min is None else x_min,
                         y_min=self.y_min if y_min is None else y_min,
                         xy_res=self.xy_res if xy_res is None else xy_res,
                         is_geo_crs=self.is_geo_crs if is_geo_crs is None else is_geo_crs)

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
        """Guess whether the x-axis crosses the antimeridian. Works currently only for geographical coordinates."""
        return self.is_geo_crs and self.x_min + self.width * self.xy_res > 180.0

    @property
    def x_min(self) -> float:
        return self._x_min

    @property
    def x_max(self) -> float:
        x_max = self.x_min + self.xy_res * self.width
        return x_max - 360.0 if self.is_geo_crs and x_max > 180.0 else x_max

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
    def is_geo_crs(self) -> bool:
        return self._is_geo_crs

    @property
    def xy_bbox(self) -> Tuple[float, float, float, float]:
        return self.x_min, self.y_min, self.x_max, self.y_max

    @property
    def xy_bboxes(self) -> np.ndarray:
        xy_offset = np.array([self.x_min, self.y_min, self.x_min, self.y_min])
        return xy_offset + self.xy_res * self.ij_bboxes

    @property
    def ij_bboxes(self) -> np.ndarray:
        chunk_sizes = get_chunk_sizes((self.height, self.width),
                                      (self.tile_height, self.tile_width))
        _, _, block_slices = get_block_iterators(chunk_sizes)
        block_slices = tuple(block_slices)
        n = len(block_slices)
        ij_bboxes = np.ndarray((n, 4), dtype=np.int64)
        for i in range(n):
            y_slice, x_slice = block_slices[i]
            ij_bboxes[i, 0] = x_slice.start
            ij_bboxes[i, 1] = y_slice.start
            ij_bboxes[i, 2] = x_slice.stop - 1
            ij_bboxes[i, 3] = y_slice.stop - 1
        return ij_bboxes

    def coord_vars(self,
                   xy_names: Tuple[str, str],
                   is_lon_normalized: bool = False,
                   is_y_axis_inverted: bool = False) -> Mapping[str, xr.DataArray]:
        x_name, y_name = xy_names
        x_attrs, y_attrs = (_LON_ATTRS, _LAT_ATTRS) if self.is_geo_crs else ({}, {})
        w, h = self.size
        x1, y1, x2, y2 = self.xy_bbox
        res = self.xy_res
        res05 = self.xy_res / 2

        x_data = np.linspace(x1 + res05, x2 - res05, w)
        x_bnds_0_data = np.linspace(x1, x2 - res, w)
        x_bnds_1_data = np.linspace(x1 + res, x2, w)

        if is_lon_normalized:
            x_data = denormalize_lon(x_data)
            x_bnds_0_data = denormalize_lon(x_bnds_0_data)
            x_bnds_1_data = denormalize_lon(x_bnds_1_data)

        y_data = np.linspace(y1 + res05, y2 - res05, h)
        y_bnds_0_data = np.linspace(y1, y2 - res, h)
        y_bnds_1_data = np.linspace(y1 + res, y2, h)

        if is_y_axis_inverted:
            y_data = y_data[::-1]
            y_bnds_1_data, y_bnds_0_data = y_bnds_0_data[::-1], y_bnds_1_data[::-1]

        bnds_name = 'bnds'
        x_bnds_name = f'{x_name}_{bnds_name}'
        y_bnds_name = f'{y_name}_{bnds_name}'

        return {
            x_name: xr.DataArray(x_data, dims=x_name, attrs=dict(**x_attrs, bounds=x_bnds_name)),
            y_name: xr.DataArray(y_data, dims=y_name, attrs=dict(**y_attrs, bounds=y_bnds_name)),
            x_bnds_name: xr.DataArray(list(zip(x_bnds_0_data, x_bnds_1_data)), dims=[x_name, bnds_name], attrs=x_attrs),
            y_bnds_name: xr.DataArray(list(zip(y_bnds_0_data, y_bnds_1_data)), dims=[y_name, bnds_name], attrs=y_attrs),
        }

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
                     xy_res=src_xy_res,
                     is_geo_crs=geo_coding.is_geo_crs)
