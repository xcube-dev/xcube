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
import warnings
from typing import Tuple, Union, Mapping, Optional

import affine
import numpy as np
import pyproj
import xarray as xr

from xcube.core.geocoding import CRS_WGS84
from xcube.core.geocoding import GeoCoding
from xcube.core.geocoding import denormalize_lon
from xcube.util.dask import get_block_iterators
from xcube.util.dask import get_chunk_sizes

_LON_ATTRS = dict(
    long_name='longitude coordinate',
    standard_name='longitude',
    units='degrees_east'
)

_LAT_ATTRS = dict(
    long_name='latitude coordinate',
    standard_name='latitude',
    units='degrees_north'
)

_X_ATTRS = dict(
    long_name="x coordinate of projection",
    standard_name="projection_x_coordinate"
)

_Y_ATTRS = dict(
    long_name="y coordinate of projection",
    standard_name="projection_y_coordinate"
)

AffineTransformMatrix = Tuple[Tuple[float, float, float],
                              Tuple[float, float, float]]


class ImageGeom:

    def __init__(self,
                 size: Union[int, Tuple[int, int]],
                 tile_size: Union[int, Tuple[int, int]] = None,
                 x_min: float = 0.0,
                 y_min: float = 0.0,
                 xy_res: Union[float, Tuple[float, float]] = None,
                 is_j_axis_up: bool = False,
                 crs: pyproj.crs.CRS = None,
                 is_geo_crs: bool = None):

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

        if isinstance(xy_res, float) or isinstance(xy_res, int):
            x_res, y_res = xy_res, xy_res
        elif xy_res is not None:
            x_res, y_res = xy_res
        else:
            x_res, y_res = 1.0, 1.0

        if x_res <= 0.0 or math.isclose(x_res, 0.0) \
                or y_res <= 0.0 or math.isclose(y_res, 0.0):
            raise ValueError('invalid xy_res')

        if is_geo_crs is not None:
            warnings.warn('keyword argument "is_geo_crs" is deprecated, use "crs" instead',
                          DeprecationWarning, stacklevel=2)

        if is_geo_crs:
            if w * x_res > 360.0:
                raise ValueError('invalid size, xy_res combination')
            if x_min < -180.0 or x_min > 180.0:
                raise ValueError('invalid x_min')
            if y_min < -90.0 or y_min > 90.0 or y_min + h * y_res > 90.0:
                raise ValueError('invalid y_min')

        if is_geo_crs is not None and crs is not None and crs.is_geographic != is_geo_crs:
            raise ValueError('crs and is_geo_crs are inconsistent')

        self._size = w, h
        self._tile_size = min(w, tw) or w, min(h, th) or h
        self._x_min = x_min
        self._y_min = y_min
        self._xy_res = x_res, y_res
        self._is_j_axis_up = is_j_axis_up
        self._crs = crs if crs is not None else CRS_WGS84

    def derive(self,
               size: Tuple[int, int] = None,
               tile_size: Union[int, Tuple[int, int]] = None,
               x_min: float = None,
               y_min: float = None,
               xy_res: Union[float, Tuple[float, float]] = None,
               is_j_axis_up: bool = None,
               crs: pyproj.crs.CRS = None,
               is_geo_crs: bool = None):
        """Derive a new image geometry from given constructor arguments."""
        return ImageGeom(self.size if size is None else size,
                         tile_size=self.tile_size if tile_size is None else tile_size,
                         x_min=self.x_min if x_min is None else x_min,
                         y_min=self.y_min if y_min is None else y_min,
                         xy_res=self.xy_res if xy_res is None else xy_res,
                         is_j_axis_up=self.is_j_axis_up if is_j_axis_up is None else is_j_axis_up,
                         crs=self.crs if crs is None else crs,
                         is_geo_crs=self.is_geo_crs if is_geo_crs is None else is_geo_crs)

    @property
    def size(self) -> Tuple[int, int]:
        """Image size (width, height) in pixels."""
        return self._size

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self._size[0]

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self._size[1]

    @property
    def is_tiled(self) -> bool:
        """Whether the image is tiled."""
        return self._size != self._tile_size

    @property
    def tile_size(self) -> Tuple[int, int]:
        """Image tile size (width, height) in pixels."""
        return self._tile_size

    @property
    def tile_width(self) -> int:
        """Image tile width in pixels."""
        return self._tile_size[0]

    @property
    def tile_height(self) -> int:
        """Image tile height in pixels."""
        return self._tile_size[1]

    @property
    def is_crossing_antimeridian(self) -> Optional[bool]:
        """
        Guess whether the x-axis crosses the anti-meridian.
        Works currently only for geographical CRSes.
        """
        if not self.is_geo_crs:
            return None
        return self.x_max > 180.0

    @property
    def x_min(self) -> float:
        """Minimum x-coordinate in CRS units."""
        return self._x_min

    @property
    def x_max(self) -> float:
        """Maximum x-coordinate in CRS units."""
        # x_max = self.x_min + self.x_res * self.width
        # return x_max - 360.0 if self.is_geo_crs and x_max > 180.0 else x_max
        return self.x_min + self.x_res * self.width

    @property
    def y_min(self) -> float:
        """Minimum y-coordinate in CRS units."""
        return self._y_min

    @property
    def y_max(self) -> float:
        """Maximum y-coordinate in CRS units."""
        return self.y_min + self.y_res * self.height

    @property
    def x_res(self) -> float:
        """Pixel size in CRS units per pixel in x-direction."""
        return self._xy_res[0]

    @property
    def y_res(self) -> float:
        """Pixel size in CRS units per pixel in y-direction."""
        return self._xy_res[1]

    @property
    def xy_res(self) -> Tuple[float, float]:
        """Pixel size in x and y direction."""
        return self._xy_res

    @property
    def avg_xy_res(self) -> float:
        """Average pixel size."""
        return 0.5 * (self.x_res + self.y_res)

    @property
    def is_j_axis_up(self) -> bool:
        """
        Does the positive image j-axis point up?
        By default, the positive image j-axis points down.
        """
        return self._is_j_axis_up

    @property
    def ij_to_xy_transform(self) -> AffineTransformMatrix:
        """The affine transformation matrix from image to CRS coordinates."""
        if self.is_j_axis_up:
            return (
                (self.x_res, 0.0, self.x_min),
                (0.0, self.y_res, self.y_min),
            )
        else:
            return (
                (self.x_res, 0.0, self.x_min),
                (0.0, -self.y_res, self.y_max),
            )

    @property
    def xy_to_ij_transform(self) -> AffineTransformMatrix:
        """The affine transformation matrix from CRS to image coordinates."""
        return ImageGeom._from_affine(~ImageGeom._to_affine(self.ij_to_xy_transform))

    @property
    def crs(self) -> pyproj.crs.CRS:
        """The coordinate reference system."""
        return self._crs

    @property
    def is_geo_crs(self) -> bool:
        """Whether the coordinate reference system is geographic."""
        return self._crs.is_geographic

    @property
    def xy_bbox(self) -> Tuple[float, float, float, float]:
        """The image's bounding box in CRS coordinates."""
        return self.x_min, self.y_min, self.x_max, self.y_max

    @property
    def xy_bboxes(self) -> np.ndarray:
        """The image chunks' bounding boxes in CRS coordinates."""
        if self.is_j_axis_up:
            xy_offset = np.array([self.x_min, self.y_min, self.x_min, self.y_min])
            xy_scale = np.array([self.x_res, self.y_res, self.x_res, self.y_res])
            xy_bboxes = xy_offset + xy_scale * self.ij_bboxes
        else:
            xy_offset = np.array([self.x_min, self.y_max, self.x_min, self.y_max])
            xy_scale = np.array([self.x_res, -self.y_res, self.x_res, -self.y_res])
            xy_bboxes = xy_offset + xy_scale * self.ij_bboxes
            xy_bboxes[:, [1, 3]] = xy_bboxes[:, [3, 1]]
        return xy_bboxes

    @property
    def ij_bboxes(self) -> np.ndarray:
        """The image chunks' bounding boxes in image pixel coordinates."""
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

    def ij_transform_from(self, other: 'ImageGeom') -> AffineTransformMatrix:
        """
        Get the affine transformation matrix that transforms
        image coordinates of *other* into image coordinates of this image geometry.

        :param other: The other image geometry
        :return: Affine transformation matrix
        """
        a = ImageGeom._to_affine(other.ij_to_xy_transform)
        b = ImageGeom._to_affine(self.xy_to_ij_transform)
        return ImageGeom._from_affine(b * a)

    def ij_transform_to(self, other: 'ImageGeom') -> AffineTransformMatrix:
        """
        Get the affine transformation matrix that transforms
        image coordinates of this image geometry to image coordinates of *other*.

        :param other: The other image geometry
        :return: Affine transformation matrix
        """
        a = ImageGeom._to_affine(self.ij_transform_from(other))
        return ImageGeom._from_affine(~a)

    @staticmethod
    def _from_affine(matrix: affine.Affine) -> AffineTransformMatrix:
        return (matrix.a, matrix.b, matrix.c), (matrix.d, matrix.e, matrix.f)

    @staticmethod
    def _to_affine(matrix: AffineTransformMatrix) -> affine.Affine:
        return affine.Affine(*matrix[0], *matrix[1])

    def coord_vars(self, xy_names: Tuple[str, str]) -> Mapping[str, xr.DataArray]:
        x_name, y_name = xy_names
        x_attrs, y_attrs = (_LON_ATTRS, _LAT_ATTRS) if self.is_geo_crs else (_X_ATTRS, _Y_ATTRS)
        w, h = self.size
        x1, y1, x2, y2 = self.xy_bbox
        x_res, y_res = self.xy_res
        x_res05 = x_res / 2
        y_res05 = y_res / 2

        x_data = np.linspace(x1 + x_res05, x2 - x_res05, w)
        x_bnds_0_data = np.linspace(x1, x2 - x_res, w)
        x_bnds_1_data = np.linspace(x1 + x_res, x2, w)

        if self.is_crossing_antimeridian:
            x_data = denormalize_lon(x_data)
            x_bnds_0_data = denormalize_lon(x_bnds_0_data)
            x_bnds_1_data = denormalize_lon(x_bnds_1_data)

        if self.is_j_axis_up:
            y_data = np.linspace(y1 + y_res05, y2 - y_res05, h)
            y_bnds_0_data = np.linspace(y1, y2 - y_res, h)
            y_bnds_1_data = np.linspace(y1 + y_res, y2, h)
        else:
            y_data = np.linspace(y2 - y_res05, y1 + y_res05, h)
            y_bnds_0_data = np.linspace(y2 - y_res, y1, h)
            y_bnds_1_data = np.linspace(y2, y1 + y_res, h)

        bnds_name = 'bnds'
        x_bnds_name = f'{x_name}_{bnds_name}'
        y_bnds_name = f'{y_name}_{bnds_name}'

        return {
            x_name: xr.DataArray(x_data,
                                 dims=x_name, attrs=dict(**x_attrs, bounds=x_bnds_name)),
            y_name: xr.DataArray(y_data,
                                 dims=y_name, attrs=dict(**y_attrs, bounds=y_bnds_name)),
            x_bnds_name: xr.DataArray(list(zip(x_bnds_0_data, x_bnds_1_data)),
                                      dims=[x_name, bnds_name], attrs=x_attrs),
            y_bnds_name: xr.DataArray(list(zip(y_bnds_0_data, y_bnds_1_data)),
                                      dims=[y_name, bnds_name], attrs=y_attrs),
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
        return _compute_image_geom(dataset,
                                   geo_coding=geo_coding,
                                   xy_names=xy_names,
                                   xy_oversampling=xy_oversampling,
                                   xy_eps=xy_eps,
                                   ij_denom=ij_denom)


def _compute_image_geom(dataset: xr.Dataset,
                        geo_coding: GeoCoding = None,
                        xy_names: Tuple[str, str] = None,
                        xy_oversampling: float = 1.0,
                        xy_eps: float = 1e-10,
                        ij_denom: Union[int, Tuple[int, int]] = None) -> ImageGeom:
    i_denom, j_denom = ((ij_denom, ij_denom) if isinstance(ij_denom, int)
                        else ij_denom) or (1, 1)
    geo_coding = geo_coding if geo_coding is not None \
        else GeoCoding.from_dataset(dataset, xy_names=xy_names)
    # TODO: make use of the new property geo_coding.is_rectified == True. In this case
    #       computations are much simpler.
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

    # Reduce height if it takes the maximum latitude over 90° (see Issue #303).
    if src_y_min + dst_height * src_xy_res > 90.0:
        dst_height -= 1

    return ImageGeom((dst_width, dst_height),
                     x_min=src_x_min,
                     y_min=src_y_min,
                     xy_res=src_xy_res,
                     crs=geo_coding.crs)
