# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
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

import abc
import copy
from typing import Mapping, Any
from typing import Tuple, Optional, Union

import numpy as np
import pyproj
import xarray as xr

from xcube.util.assertions import assert_condition
from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_instance
from xcube.util.dask import get_block_iterators
from xcube.util.dask import get_chunk_sizes
from .helpers import AffineTransformMatrix
from .helpers import Number
from .helpers import _from_affine
from .helpers import _parse_int_pair
from .helpers import _parse_number_pair
from .helpers import _to_affine
from .helpers import from_lon_360

# WGS84, axis order: lat, lon
CRS_WGS84 = pyproj.crs.CRS(4326)

# WGS84, axis order: lon, lat
CRS_CRS84 = pyproj.crs.CRS.from_string("urn:ogc:def:crs:OGC:1.3:CRS84")


class GridMapping(abc.ABC):
    """
    An abstract base class for grid mappings that define an image grid and
    a transformation from image pixel coordinates to spatial Earth coordinates
    defined in a well-known coordinate reference system (CRS).

    This class cannot be instantiated directly. Use one of its factory methods
    the create instances:

    * :meth:from_min_res()
    * :meth:from_dataset()
    * :meth:from_coords()
    * :meth:transform()

    """

    def __init__(self,
                 /,
                 size: Union[int, Tuple[int, int]],
                 tile_size: Optional[Union[int, Tuple[int, int]]],
                 xy_bbox: Tuple[Number, Number, Number, Number],
                 xy_res: Union[Number, Tuple[Number, Number]],
                 crs: pyproj.crs.CRS,
                 is_regular: Optional[bool],
                 is_lon_360: Optional[bool],
                 is_j_axis_up: Optional[bool]):

        width, height = _parse_int_pair(size, name='size')
        assert_condition(width > 1 and height > 1, 'invalid size')

        tile_width, tile_height = _parse_int_pair(tile_size, default=(width, height))
        assert_condition(tile_width > 1 and tile_height > 1, 'invalid tile_size')

        assert_given(xy_bbox, 'xy_bbox')
        assert_given(xy_res, 'xy_res')
        assert_instance(crs, pyproj.crs.CRS)

        x_min, y_min, x_max, y_max = xy_bbox
        x_res, y_res = _parse_number_pair(xy_res, name='xy_res')
        assert_condition(x_res > 0 and y_res > 0, 'invalid xy_res')

        self._size = width, height
        self._tile_size = tile_width, tile_height
        self._xy_bbox = x_min, y_min, x_max, y_max
        self._xy_res = x_res, y_res
        self._crs = crs
        self._is_regular = is_regular
        self._is_lon_360 = is_lon_360
        self._is_j_axis_up = is_j_axis_up

    def derive(self,
               /,
               tile_size: Union[int, Tuple[int, int]] = None,
               is_j_axis_up: bool = None):
        other = copy.copy(self)
        if tile_size is not None:
            tile_width, tile_height = _parse_int_pair(tile_size, name='tile_size')
            assert_condition(tile_width > 1 and tile_height > 1, 'invalid tile_size')
            other._tile_size = tile_width, tile_height
        if is_j_axis_up is not None:
            other._is_j_axis_up = is_j_axis_up
        return other

    @property
    def size(self) -> Tuple[int, int]:
        """Image size (width, height) in pixels."""
        return self._size

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self.size[0]

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self.size[1]

    @property
    def tile_size(self) -> Tuple[int, int]:
        """Image tile size (width, height) in pixels."""
        return self._tile_size

    @property
    def is_tiled(self) -> bool:
        """Whether the image is tiled."""
        return self.size != self.tile_size

    @property
    def tile_width(self) -> int:
        """Image tile width in pixels."""
        return self.tile_size[0]

    @property
    def tile_height(self) -> int:
        """Image tile height in pixels."""
        return self.tile_size[1]

    @property
    @abc.abstractmethod
    def xy_coords(self) -> xr.DataArray:
        """
        The x,y coordinates as data array of shape (2, height, width).
        Coordinates are given in units of the CRS.
        """

    @property
    def xy_bbox(self) -> Tuple[float, float, float, float]:
        """The image's bounding box in CRS coordinates."""
        return self._xy_bbox

    @property
    def x_min(self) -> Number:
        """Minimum x-coordinate in CRS units."""
        return self._xy_bbox[0]

    @property
    def y_min(self) -> Number:
        """Minimum y-coordinate in CRS units."""
        return self._xy_bbox[1]

    @property
    def x_max(self) -> Number:
        """Maximum x-coordinate in CRS units."""
        return self._xy_bbox[2]

    @property
    def y_max(self) -> Number:
        """Maximum y-coordinate in CRS units."""
        return self._xy_bbox[3]

    @property
    def xy_res(self) -> Tuple[Number, Number]:
        """Pixel size in x and y direction."""
        return self._xy_res

    @property
    def x_res(self) -> Number:
        """Pixel size in CRS units per pixel in x-direction."""
        return self._xy_res[0]

    @property
    def y_res(self) -> Number:
        """Pixel size in CRS units per pixel in y-direction."""
        return self._xy_res[1]

    @property
    def crs(self) -> pyproj.crs.CRS:
        """The coordinate reference system."""
        return self._crs

    @property
    def is_lon_360(self) -> Optional[bool]:
        """
        Check whether *x_max* is greater than 180 degrees.
        Effectively tests whether the range *x_min*, *x_max* crosses
        the anti-meridian at 180 degrees.
        Works only for geographical coordinate reference systems.
        """
        return self._is_lon_360

    @property
    def is_regular(self) -> Optional[bool]:
        """
        Do the x,y coordinates for a regular grid?
        A regular grid has a constant delta in both x- and y-directions of the x- and y-coordinates.
        :return None, if this property cannot be determined, True or False otherwise.
        """
        return self._is_regular

    @property
    def is_j_axis_up(self) -> Optional[bool]:
        """
        Does the positive image j-axis point up?
        By default, the positive image j-axis points down.
         :return None, if this property cannot be determined, True or False otherwise.
        """
        return self._is_j_axis_up

    @property
    def ij_to_xy_transform(self) -> AffineTransformMatrix:
        """
        The affine transformation matrix from image to CRS coordinates.
        Defined only for grid mappings with rectified x,y coordinates.
        """
        self._assert_regular()
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
        """
        The affine transformation matrix from CRS to image coordinates.
        Defined only for grid mappings with rectified x,y coordinates.
        """
        self._assert_regular()
        return _from_affine(~_to_affine(self.ij_to_xy_transform))

    def ij_transform_from(self, other: 'GridMapping') -> AffineTransformMatrix:
        """
        Get the affine transformation matrix that transforms
        image coordinates of *other* into image coordinates of this image geometry.

        Defined only for grid mappings with rectified x,y coordinates.

        :param other: The other image geometry
        :return: Affine transformation matrix
        """
        self._assert_regular()
        _assert_regular_grid_mapping(other, name='other')
        a = _to_affine(other.ij_to_xy_transform)
        b = _to_affine(self.xy_to_ij_transform)
        return _from_affine(b * a)

    def ij_transform_to(self, other: 'GridMapping') -> AffineTransformMatrix:
        """
        Get the affine transformation matrix that transforms
        image coordinates of this image geometry to image coordinates of *other*.

        Defined only for grid mappings with rectified x,y coordinates.

        :param other: The other image geometry
        :return: Affine transformation matrix
        """
        self._assert_regular()
        _assert_regular_grid_mapping(other, name='other')
        a = _to_affine(self.ij_transform_from(other))
        return _from_affine(~a)

    @property
    def ij_bbox(self) -> Tuple[int, int, int, int]:
        """The image's bounding box in pixel coordinates."""
        return 0, 0, self.width, self.height

    @property
    def ij_bboxes(self) -> np.ndarray:
        """The image tiles' bounding boxes in image pixel coordinates."""
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

    @property
    def xy_bboxes(self) -> np.ndarray:
        """The image tiles' bounding boxes in CRS coordinates."""
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

    def ij_bbox_from_xy_bbox(self,
                             xy_bbox: Tuple[float, float, float, float],
                             xy_border: float = 0.0,
                             ij_border: int = 0) -> Tuple[int, int, int, int]:
        """
        Compute bounding box in i,j pixel coordinates given a bounding box *xy_bbox* in x,y coordinates.

        :param xy_bbox: Box (x_min, y_min, x_max, y_max) given in the same CS as x and y.
        :param xy_border: If non-zero, grows the bounding box *xy_bbox* before using it for comparisons. Defaults to 0.
        :param ij_border: If non-zero, grows the returned i,j bounding box and clips it to size. Defaults to 0.
        :return: Bounding box in (i_min, j_min, i_max, j_max) in pixel coordinates.
            Returns ``(-1, -1, -1, -1)`` if *xy_bbox* isn't intersecting any of the x,y coordinates.
        """
        xy_bboxes = np.array([xy_bbox], dtype=np.float64)
        ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        self.ij_bboxes_from_xy_bboxes(xy_bboxes, xy_border=xy_border, ij_border=ij_border, ij_bboxes=ij_bboxes)
        # noinspection PyTypeChecker
        return tuple(map(int, ij_bboxes[0]))

    def ij_bboxes_from_xy_bboxes(self,
                                 xy_bboxes: np.ndarray,
                                 xy_border: float = 0.0,
                                 ij_border: int = 0,
                                 ij_bboxes: np.ndarray = None) -> np.ndarray:
        """
        Compute bounding boxes in i,j pixel coordinates given bounding boxes *xy_bboxes* in x,y coordinates.

        :param xy_bboxes: Numpy array of x,y bounding boxes [[x_min, y_min, x_max, y_max], ...]
            given in the same CS as x and y.
        :param xy_border: If non-zero, grows the bounding box *xy_bbox* before using it for comparisons. Defaults to 0.
        :param ij_border: If non-zero, grows the returned i,j bounding box and clips it to size. Defaults to 0.
        :param ij_bboxes: Numpy array of pixel i,j bounding boxes [[x_min, y_min, x_max, y_max], ...].
            If given, must have same shape as *xy_bboxes*.
        :return: Bounding box in (i_min, j_min, i_max, j_max) in pixel coordinates.
            Returns None if *xy_bbox* isn't intersecting any of the x,y coordinates.
        """
        from .bboxes import compute_ij_boxes
        if ij_bboxes is None:
            ij_bboxes = np.full_like(xy_bboxes, -1, dtype=np.int64)
        else:
            ij_bboxes[:, :] = -1
        xy_coords = self.xy_coords.values
        compute_ij_boxes(xy_coords[0],
                         xy_coords[1],
                         xy_bboxes,
                         xy_border,
                         ij_border,
                         ij_bboxes)
        return ij_bboxes

    def coord_vars(self, xy_names: Tuple[str, str]) -> Mapping[str, xr.DataArray]:
        """
        Get CF-compliant axis coordinate variables and cell boundary coordinate variables.

        Defined only for grid mappings with rectified x,y coordinates.

        :param xy_names:
        :return: dictionary with coordinate variables
        """
        self._assert_regular()
        x_name, y_name = xy_names
        w, h = self.size
        x1, y1, x2, y2 = self.xy_bbox
        x_res, y_res = self.xy_res
        x_res05 = x_res / 2
        y_res05 = y_res / 2

        dtype = np.float64

        x_data = np.linspace(x1 + x_res05, x2 - x_res05, w, dtype=dtype)
        x_bnds_0_data = np.linspace(x1, x2 - x_res, w, dtype=dtype)
        x_bnds_1_data = np.linspace(x1 + x_res, x2, w, dtype=dtype)

        if self.is_lon_360:
            x_data = from_lon_360(x_data)
            x_bnds_0_data = from_lon_360(x_bnds_0_data)
            x_bnds_1_data = from_lon_360(x_bnds_1_data)

        if self.is_j_axis_up:
            y_data = np.linspace(y1 + y_res05, y2 - y_res05, h, dtype=dtype)
            y_bnds_0_data = np.linspace(y1, y2 - y_res, h, dtype=dtype)
            y_bnds_1_data = np.linspace(y1 + y_res, y2, h, dtype=dtype)
        else:
            y_data = np.linspace(y2 - y_res05, y1 + y_res05, h, dtype=dtype)
            y_bnds_0_data = np.linspace(y2 - y_res, y1, h, dtype=dtype)
            y_bnds_1_data = np.linspace(y2, y1 + y_res, h, dtype=dtype)

        bnds_name = 'bnds'
        x_bnds_name = f'{x_name}_{bnds_name}'
        y_bnds_name = f'{y_name}_{bnds_name}'

        if self.crs.is_geographic:
            x_attrs = dict(
                long_name='longitude coordinate',
                standard_name='longitude',
                units='degrees_east'
            )
            y_attrs = dict(
                long_name='latitude coordinate',
                standard_name='latitude',
                units='degrees_north'
            )
        else:
            x_attrs = dict(
                long_name="x coordinate of projection",
                standard_name="projection_x_coordinate"
            )
            y_attrs = dict(
                long_name="y coordinate of projection",
                standard_name="projection_y_coordinate"
            )

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

    def transform(self,
                  target_crs: pyproj.crs.CRS,
                  *,
                  tile_size: Union[int, Tuple[int, int]] = None) -> 'GridMapping':
        from .transform import transform_grid_mapping
        return transform_grid_mapping(self, target_crs, tile_size=tile_size)

    @classmethod
    def from_min_res(cls,
                     size: Union[int, Tuple[int, int]],
                     xy_min: Tuple[float, float],
                     xy_res: Union[float, Tuple[float, float]],
                     crs: pyproj.crs.CRS,
                     *,
                     tile_size: Union[int, Tuple[int, int]] = None,
                     is_j_axis_up: bool = False) -> 'GridMapping':
        from .regular import from_min_res
        return from_min_res(size=size,
                            xy_min=xy_min,
                            xy_res=xy_res,
                            crs=crs,
                            tile_size=tile_size,
                            is_j_axis_up=is_j_axis_up)

    @classmethod
    def from_dataset(cls,
                     dataset: xr.Dataset,
                     *,
                     tile_size: Union[int, Tuple[int, int]] = None,
                     prefer_regular: bool = True,
                     prefer_crs: pyproj.crs.CRS = None,
                     emit_warnings: bool = False) -> 'GridMapping':
        from .dataset import from_dataset
        return from_dataset(dataset=dataset,
                            tile_size=tile_size,
                            prefer_regular=prefer_regular,
                            prefer_crs=prefer_crs,
                            emit_warnings=emit_warnings)

    @classmethod
    def from_coords(cls,
                    x_coords: xr.DataArray,
                    y_coords: xr.DataArray,
                    crs: pyproj.crs.CRS,
                    *,
                    tile_size: Union[int, Tuple[int, int]] = None) -> 'GridMapping':
        from .coords import from_coords
        return from_coords(x_coords=x_coords, y_coords=y_coords, crs=crs, tile_size=tile_size)

    def _assert_regular(self):
        if not self.is_regular:
            raise NotImplementedError('Operation not implemented for non-regular grid mappings')


def _assert_regular_grid_mapping(value: Any, name: str = None):
    assert_instance(value, GridMapping, name=name)
    if not value.is_regular:
        raise ValueError(f'{name or "value"} must be a regular grid mapping')

# def _assert_valid_xy_coords(xy_coords: Any):
#     assert_instance(xy_coords, xr.DataArray, name='xy_coords')
#     assert_condition(xy_coords.ndim == 3
#                      and xy_coords.shape[0] == 2
#                      and xy_coords.shape[1] >= 2
#                      and xy_coords.shape[2] >= 2,
#                      'xy_coords must have dimensions (2, height, width) with height >= 2 and width >= 2')
