# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
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
from typing import Optional, Union, Dict, Tuple, Sequence, Any

import affine
import numpy as np
import rasterio.features
import shapely.geometry
import shapely.geometry
import shapely.wkt
import xarray as xr

from .geojson import GeoJSON
from .update import update_dataset_spatial_attrs

GeometryLike = Union[shapely.geometry.base.BaseGeometry, Dict[str, Any], str, Sequence[Union[float, int]]]
Bounds = Tuple[float, float, float, float]
SplitBounds = Tuple[Bounds, Optional[Bounds]]

_INVALID_GEOMETRY_MSG = ('Geometry must be either a shapely geometry object, '
                         'a GeoJSON-serializable dictionary, a geometry WKT string, '
                         'box coordinates (x1, y1, x2, y2), '
                         'or point coordinates (x, y)')

_INVALID_BOX_COORDS_MSG = 'Invalid box coordinates'


def mask_dataset_by_geometry(dataset: xr.Dataset,
                             geometry: GeometryLike,
                             no_clip: bool = False,
                             save_geometry_mask: Union[str, bool] = False,
                             save_geometry_wkt: Union[str, bool] = False) -> Optional[xr.Dataset]:
    """
    Mask a dataset according to the given geometry. The cells of variables of the
    returned dataset will have NaN-values where their spatial coordinates are not intersecting
    the given geometry.

    :param dataset: The dataset
    :param geometry: A geometry-like object, see py:function:`convert_geometry`.
    :param no_clip: If True, the function will not clip the dataset before masking, this is, the
        returned dataset will have the same dimension size as the given *dataset*.
    :param save_geometry_mask: If the value is a string, the effective geometry mask array is stored as
        a 2D data variable named by *save_geometry_mask*.
        If the value is True, the name "geometry_mask" is used.
    :param save_geometry_wkt: If the value is a string, the effective intersection geometry is stored as
        a Geometry WKT string in the global attribute named by *save_geometry*.
        If the value is True, the name "geometry_wkt" is used.
    :return: The dataset spatial subset, or None if the bounding box of the dataset has a no or a zero area
        intersection with the bounding box of the geometry.
    """
    geometry = convert_geometry(geometry)

    intersection_geometry = intersect_geometries(get_dataset_bounds(dataset), geometry)
    if intersection_geometry is None:
        return None

    if not no_clip:
        dataset = _clip_dataset_by_geometry(dataset, intersection_geometry)

    ds_x_min, ds_y_min, ds_x_max, ds_y_max = get_dataset_bounds(dataset)

    width = dataset.dims['lon']
    height = dataset.dims['lat']
    spatial_res = (ds_x_max - ds_x_min) / width

    mask_data = get_geometry_mask(width, height, intersection_geometry, ds_x_min, ds_y_min, spatial_res)
    mask = xr.DataArray(mask_data,
                        coords=dict(lat=dataset.lat, lon=dataset.lon),
                        dims=('lat', 'lon'))

    masked_vars = {}
    for var_name in dataset.data_vars:
        var = dataset[var_name]
        masked_vars[var_name] = var.where(mask)

    masked_dataset = xr.Dataset(masked_vars, coords=dataset.coords, attrs=dataset.attrs)

    _save_geometry_mask(masked_dataset, mask, save_geometry_mask)
    _save_geometry_wkt(masked_dataset, intersection_geometry, save_geometry_wkt)

    return masked_dataset


def clip_dataset_by_geometry(dataset: xr.Dataset,
                             geometry: GeometryLike,
                             save_geometry_wkt: Union[str, bool] = False) -> Optional[xr.Dataset]:
    """
    Spatially clip a dataset according to the bounding box of a given geometry.

    :param dataset: The dataset
    :param geometry: A geometry-like object, see py:function:`convert_geometry`.
    :param save_geometry_wkt: If the value is a string, the effective intersection geometry is stored as
        a Geometry WKT string in the global attribute named by *save_geometry*.
        If the value is True, the name "geometry_wkt" is used.
    :return: The dataset spatial subset, or None if the bounding box of the dataset has a no or a zero area
        intersection with the bounding box of the geometry.
    """
    intersection_geometry = intersect_geometries(get_dataset_bounds(dataset), geometry)
    if intersection_geometry is None:
        return None
    return _clip_dataset_by_geometry(dataset, intersection_geometry, save_geometry_wkt=save_geometry_wkt)


def _clip_dataset_by_geometry(dataset: xr.Dataset,
                              intersection_geometry: shapely.geometry.base.BaseGeometry,
                              save_geometry_wkt: bool = False) -> Optional[xr.Dataset]:
    # TODO (forman): the following code is wrong, if the dataset bounds cross the anti-meridian!

    ds_x_min, ds_y_min, ds_x_max, ds_y_max = get_dataset_bounds(dataset)

    width = dataset.lon.size
    height = dataset.lat.size
    res = (ds_y_max - ds_y_min) / height

    g_lon_min, g_lat_min, g_lon_max, g_lat_max = intersection_geometry.bounds
    x1 = _clamp(int(math.floor((g_lon_min - ds_x_min) / res)), 0, width - 1)
    x2 = _clamp(int(math.ceil((g_lon_max - ds_x_min) / res)), 0, width - 1)
    y1 = _clamp(int(math.floor((g_lat_min - ds_y_min) / res)), 0, height - 1)
    y2 = _clamp(int(math.ceil((g_lat_max - ds_y_min) / res)), 0, height - 1)
    if not is_dataset_y_axis_inverted(dataset):
        _y1, _y2 = y1, y2
        y1 = height - _y2 - 1
        y2 = height - _y1 - 1

    dataset_subset = dataset.isel(lon=slice(x1, x2), lat=slice(y1, y2))

    update_dataset_spatial_attrs(dataset_subset, update_existing=True, in_place=True)

    _save_geometry_wkt(dataset_subset, intersection_geometry, save_geometry_wkt)

    return dataset_subset


def _save_geometry_mask(dataset, mask, save_mask):
    if save_mask:
        var_name = save_mask if isinstance(save_mask, str) else 'geometry_mask'
        dataset[var_name] = mask


def _save_geometry_wkt(dataset, intersection_geometry, save_geometry):
    if save_geometry:
        attr_name = save_geometry if isinstance(save_geometry, str) else 'geometry_wkt'
        dataset.attrs.update({attr_name: intersection_geometry.wkt})


def get_geometry_mask(width: int, height: int,
                      geometry: GeometryLike,
                      lon_min: float, lat_min: float, res: float) -> np.ndarray:
    geometry = convert_geometry(geometry)
    # noinspection PyTypeChecker
    transform = affine.Affine(res, 0.0, lon_min,
                              0.0, -res, lat_min + res * height)
    return rasterio.features.geometry_mask([geometry],
                                           out_shape=(height, width),
                                           transform=transform,
                                           all_touched=True,
                                           invert=True)


def intersect_geometries(geometry1: GeometryLike, geometry2: GeometryLike) \
        -> Optional[shapely.geometry.base.BaseGeometry]:
    geometry1 = convert_geometry(geometry1)
    if geometry1 is None:
        return None
    geometry2 = convert_geometry(geometry2)
    if geometry2 is None:
        return geometry1
    intersection_geometry = geometry1.intersection(geometry2)
    if not intersection_geometry.is_valid or intersection_geometry.is_empty:
        return None
    return intersection_geometry


def convert_geometry(geometry: Optional[GeometryLike]) -> Optional[shapely.geometry.base.BaseGeometry]:
    """
    Convert a geometry-like object into a shapely geometry object (``shapely.geometry.BaseGeometry``).

    A geometry-like object is may be any shapely geometry object,
    * a dictionary that can be serialized to valid GeoJSON,
    * a WKT string,
    * a box given by a string of the form "<x1>,<y1>,<x2>,<y2>"
      or by a sequence of four numbers x1, y1, x2, y2,
    * a point by a string of the form "<x>,<y>"
      or by a sequence of two numbers x, y.

    Handling of geometries crossing the antimeridian:

    * If box coordinates are given, it is allowed to pass x1, x2 where x1 > x2,
      which is interpreted as a box crossing the antimeridian. In this case the function
      splits the box along the antimeridian and returns a multi-polygon.
    * In all other cases, 2D geometries are assumed to _not cross the antimeridian at all_.

    :param geometry: A geometry-like object
    :return:  Shapely geometry object or None.
    """

    if isinstance(geometry, shapely.geometry.base.BaseGeometry):
        return geometry

    if isinstance(geometry, dict):
        if GeoJSON.is_geometry(geometry):
            return shapely.geometry.shape(geometry)
        elif GeoJSON.is_feature(geometry):
            geometry = GeoJSON.get_feature_geometry(geometry)
            if geometry is not None:
                return shapely.geometry.shape(geometry)
        elif GeoJSON.is_feature_collection(geometry):
            features = GeoJSON.get_feature_collection_features(geometry)
            if features is not None:
                geometries = [f2 for f2 in [GeoJSON.get_feature_geometry(f1) for f1 in features] if f2 is not None]
                if geometries:
                    geometry = dict(type='GeometryCollection', geometries=geometries)
                    return shapely.geometry.shape(geometry)
        raise ValueError(_INVALID_GEOMETRY_MSG)

    if isinstance(geometry, str):
        return shapely.wkt.loads(geometry)

    if geometry is None:
        return None

    invalid_box_coords = False
    # noinspection PyBroadException
    try:
        x1, y1, x2, y2 = geometry
        is_point = x1 == x2 and y1 == y2
        if is_point:
            return shapely.geometry.Point(x1, y1)
        invalid_box_coords = x1 == x2 or y1 >= y2
        if not invalid_box_coords:
            return get_box_split_bounds_geometry(x1, y1, x2, y2)
    except Exception:
        # noinspection PyBroadException
        try:
            x, y = geometry
            return shapely.geometry.Point(x, y)
        except Exception:
            pass

    if invalid_box_coords:
        raise ValueError(_INVALID_BOX_COORDS_MSG)
    raise ValueError(_INVALID_GEOMETRY_MSG)


def is_dataset_y_axis_inverted(dataset: Union[xr.Dataset, xr.DataArray]) -> bool:
    if 'lat' in dataset.coords:
        y = dataset.lat
    elif 'y' in dataset.coords:
        y = dataset.y
    else:
        raise ValueError("Neither 'lat' nor 'y' coordinate variable found.")
    return float(y[0]) < float(y[-1])


def get_dataset_geometry(dataset: Union[xr.Dataset, xr.DataArray]) -> shapely.geometry.base.BaseGeometry:
    return get_box_split_bounds_geometry(*get_dataset_bounds(dataset))


def get_dataset_bounds(dataset: Union[xr.Dataset, xr.DataArray]) -> Bounds:
    lon_var = dataset.coords.get("lon")
    lat_var = dataset.coords.get("lat")
    if lon_var is None:
        raise ValueError('Missing coordinate variable "lon"')
    if lat_var is None:
        raise ValueError('Missing coordinate variable "lat"')

    lon_bnds_name = lon_var.attrs["bounds"] if "bounds" in lon_var.attrs else "lon_bnds"
    if lon_bnds_name in dataset.coords:
        lon_bnds_var = dataset.coords[lon_bnds_name]
        lon_min = lon_bnds_var[0][0]
        lon_max = lon_bnds_var[-1][1]
    else:
        lon_min = lon_var[0]
        lon_max = lon_var[-1]
        delta = min(abs(np.diff(lon_var)))
        lon_min -= 0.5 * delta
        lon_max += 0.5 * delta

    lat_bnds_name = lat_var.attrs["bounds"] if "bounds" in lat_var.attrs else "lat_bnds"
    if lat_bnds_name in dataset.coords:
        lat_bnds_var = dataset.coords[lat_bnds_name]
        lat1 = lat_bnds_var[0][0]
        lat2 = lat_bnds_var[-1][1]
        lat_min = min(lat1, lat2)
        lat_max = max(lat1, lat2)
    else:
        lat1 = lat_var[0]
        lat2 = lat_var[-1]
        delta = min(abs(np.diff(lat_var)))
        lat_min = min(lat1, lat2) - 0.5 * delta
        lat_max = max(lat1, lat2) + 0.5 * delta

    return float(lon_min), float(lat_min), float(lon_max), float(lat_max)


def get_box_split_bounds(lon_min: float, lat_min: float,
                         lon_max: float, lat_max: float) -> SplitBounds:
    if lon_max >= lon_min:
        return (lon_min, lat_min, lon_max, lat_max), None
    else:
        return (lon_min, lat_min, 180.0, lat_max), (-180.0, lat_min, lon_max, lat_max)


def get_box_split_bounds_geometry(lon_min: float, lat_min: float,
                                  lon_max: float, lat_max: float) -> shapely.geometry.base.BaseGeometry:
    box_1, box_2 = get_box_split_bounds(lon_min, lat_min, lon_max, lat_max)
    if box_2 is not None:
        return shapely.geometry.MultiPolygon(polygons=[shapely.geometry.box(*box_1), shapely.geometry.box(*box_2)])
    else:
        return shapely.geometry.box(*box_1)


def _clamp(x, x1, x2):
    if x < x1:
        return x1
    if x > x2:
        return x2
    return x
