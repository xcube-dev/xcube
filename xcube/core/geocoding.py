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

from typing import Sequence, Tuple, Optional, Union

import numba as nb
import numpy as np
import xarray as xr

LON_COORD_VAR_NAMES = ('lon', 'long', 'longitude')
LAT_COORD_VAR_NAMES = ('lat', 'latitude')
X_COORD_VAR_NAMES = ('x', 'xc') + LON_COORD_VAR_NAMES
Y_COORD_VAR_NAMES = ('y', 'yc') + LAT_COORD_VAR_NAMES


class GeoCoding:

    # TODO (forman): add docs

    def __init__(self,
                 x: xr.DataArray,
                 y: xr.DataArray,
                 x_name: str = None,
                 y_name: str = None,
                 is_geo_crs: bool = None,
                 is_lon_normalized: bool = False):
        is_geo_crs = True if is_geo_crs is None and is_lon_normalized else bool(is_geo_crs)
        if is_lon_normalized and not is_geo_crs:
            raise ValueError('is_lon_normalized cannot be True while is_geo_crs is False')
        x_name = x_name or x.name
        y_name = y_name or y.name
        if not x_name or not y_name:
            raise ValueError('failed to determine x_name, y_name from x, y')
        # TODO (forman): validate args & kwargs
        self._x = x
        self._y = y
        self._x_name = x_name
        self._y_name = y_name
        self._is_geo_crs = is_geo_crs
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
    def is_geo_crs(self) -> bool:
        return self._is_geo_crs

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
               is_geo_crs: bool = None,
               is_lon_normalized: bool = None):
        return GeoCoding(x=x if x is not None else self.x,
                         y=y if y is not None else self.y,
                         x_name=x_name if x_name is not None else self.x_name,
                         y_name=y_name if y_name is not None else self.y_name,
                         is_geo_crs=is_geo_crs if is_geo_crs is not None
                         else self.is_geo_crs,
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

        is_geo_crs = _is_geo_crs(x_name, y_name)
        is_lon_normalized = False
        if is_geo_crs:
            x, is_lon_normalized = _maybe_normalise_2d_lon(x)

        return GeoCoding(x=x, y=y,
                         x_name=x_name,
                         y_name=y_name,
                         is_geo_crs=is_geo_crs,
                         is_lon_normalized=is_lon_normalized)

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

def _is_geo_crs(x_name: str, y_name: str) -> bool:
    return x_name in LON_COORD_VAR_NAMES and y_name in LAT_COORD_VAR_NAMES

def normalize_lon(lon_var: Union[np.ndarray, xr.DataArray]):
    if isinstance(lon_var, xr.DataArray):
        return lon_var.where(lon_var >= 0.0, lon_var + 360.0)
    else:
        return np.where(lon_var >= 0.0, lon_var, lon_var + 360.0)


def denormalize_lon(lon_var: Union[np.ndarray, xr.DataArray]):
    if isinstance(lon_var, xr.DataArray):
        return lon_var.where(lon_var <= 180.0, lon_var - 360.0)
    else:
        return np.where(lon_var <= 180.0, lon_var, lon_var - 360.0)


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


def _get_dataset_xy_names(dataset: xr.Dataset, xy_names: Tuple[str, str] = None) -> Tuple[str, str]:
    # TODO (forman): merge logic with xcube.core.schema.get_dataset_xy_var_names(dataset) and use it instead
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


def _is_crossing_antimeridian(lon_var: xr.DataArray):
    dim_y, dim_x = lon_var.dims
    # noinspection PyTypeChecker
    return abs(lon_var.diff(dim=dim_x)).max() > 180.0 or \
           abs(lon_var.diff(dim=dim_y)).max() > 180.0


def _maybe_normalise_2d_lon(lon_var: xr.DataArray):
    if _is_crossing_antimeridian(lon_var):
        lon_var = normalize_lon(lon_var)
        if _is_crossing_antimeridian(lon_var):
            raise ValueError('cannot account for longitudial anti-meridian crossing')
        return lon_var, True
    return lon_var, False
