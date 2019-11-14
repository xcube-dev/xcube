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
from typing import Optional, Union, Dict, Tuple, Sequence, Any, Mapping, Callable

import affine
import numpy as np
import rasterio.features
import shapely.geometry
import shapely.geometry
import shapely.wkt
import xarray as xr

from xcube.core.schema import get_dataset_xy_var_names, get_dataset_bounds_var_name
from xcube.core.update import update_dataset_spatial_attrs
from xcube.util.geojson import GeoJSON

GeometryLike = Union[shapely.geometry.base.BaseGeometry, Dict[str, Any], str, Sequence[Union[float, int]]]
Bounds = Tuple[float, float, float, float]
SplitBounds = Tuple[Bounds, Optional[Bounds]]

_INVALID_GEOMETRY_MSG = ('Geometry must be either a shapely geometry object, '
                         'a GeoJSON-serializable dictionary, a geometry WKT string, '
                         'box coordinates (x1, y1, x2, y2), '
                         'or point coordinates (x, y)')

_INVALID_BOX_COORDS_MSG = 'Invalid box coordinates'

Name = str
Attrs = Mapping[Name, Any]
GeoJSONFeature = Mapping[Name, Any]
GeoJSONFeatures = Sequence[GeoJSONFeature]
GeoDataFrame = 'pandas.geodataframe.GeoDataFrame'
AttrConverter = Callable[[Any], Any]
VarPropsTuple = Tuple[Name, Any, Any, Optional[Attrs], AttrConverter]


def rasterize_features(dataset: xr.Dataset,
                       features: Union[GeoDataFrame, GeoJSONFeatures],
                       feature_property_names: Sequence[Name],
                       var_properties: Dict[Name, Union[Name, VarPropsTuple]] = None,
                       in_place: bool = False) -> Optional[xr.Dataset]:
    """
    Rasterize numeric feature properties of GeoJSON *features* as new variables into *dataset*.

    Using *var_properties* the properties of newly created variables can be determined.
    If provided, it must be a mapping from feature property name to a variable name
    or a 5-tuple ``(name, dtype, fill_value, attributes, converter)`` to be used for a new variable.
    If only variable name is given, the defaults are
    ``(name, dtype='float64', fill_value=float('nan'), attributes=None, converter=float)``.

    Currently, the coordinates of the geometries in the given *features* must use the same CRS as
    the given *dataset*.

    :param dataset: The xarray dataset.
    :param features: Sequence of GeoJSON features.
    :param feature_property_names: Sequence of names of numeric feature properties to be rasterized.
    :param var_properties: Optional mapping of feature property name
        to a name or a 5-tuple (name, dtype, fill_value, attributes, converter) for the new variable.
    :param in_place: Whether to add new variables to *dataset*.
        If False, a copy will be created and returned.
    :return: dataset with rasterized feature_property
    """
    import geopandas

    var_properties = var_properties or {}
    xy_var_names = get_dataset_xy_var_names(dataset, must_exist=True)
    dataset_bounds = get_dataset_bounds(dataset, xy_var_names=xy_var_names)

    ds_x_min, ds_y_min, ds_x_max, ds_y_max = dataset_bounds

    x_var_name, y_var_name = xy_var_names
    x_var, y_var = dataset[x_var_name], dataset[y_var_name]
    x_dim, y_dim = x_var.dims[0], y_var.dims[0]
    coords = {y_var_name: y_var, x_var_name: x_var}
    dims = (y_dim, x_dim)

    width = x_var.size
    height = y_var.size
    spatial_res = (ds_x_max - ds_x_min) / width

    if geopandas and isinstance(features, geopandas.GeoDataFrame):
        geo_data_frame = features
    else:
        geo_data_frame = geopandas.GeoDataFrame.from_features(features)

    for feature_property_name in feature_property_names:
        if feature_property_name not in geo_data_frame:
            raise ValueError(f'feature property {feature_property_name!r} not found')

    if not in_place:
        dataset = xr.Dataset(coords=dataset.coords, attrs=dataset.attrs)

    for row in range(len(geo_data_frame)):
        geometry = geo_data_frame.geometry[row]
        if geometry.is_empty or not geometry.is_valid:
            continue

        # TODO (forman): allow transforming geometry into CRS of dataset here

        intersection_geometry = intersect_geometries(dataset_bounds, geometry)
        if intersection_geometry is None:
            continue

        mask_data = get_geometry_mask(width, height, intersection_geometry, ds_x_min, ds_y_min, spatial_res)
        mask = xr.DataArray(mask_data, coords=coords, dims=dims)

        for feature_property_name in feature_property_names:

            var_name = var_properties.get(feature_property_name, feature_property_name)
            var_dtype = np.float64
            var_fill_value = None
            var_attrs = None
            attr_converter = None
            if not isinstance(var_name, str):
                var_name, var_dtype, var_fill_value, var_attrs, attr_converter = var_name
            var_dtype = var_dtype if var_dtype is not None else np.float64
            var_fill_value = var_fill_value if var_fill_value is not None else np.nan
            attr_converter = attr_converter if attr_converter is not None else float

            feature_property_value = attr_converter(geo_data_frame[feature_property_name][row])

            var_new = xr.DataArray(np.full((height, width), feature_property_value, dtype=var_dtype),
                                   coords=coords, dims=dims, attrs=var_attrs)
            if var_name not in dataset:
                var_old = xr.DataArray(np.full((height, width), var_fill_value, dtype=var_dtype),
                                       coords=coords, dims=dims, attrs=var_attrs)
                dataset[var_name] = var_old
            else:
                var_old = dataset[var_name]

            dataset[var_name] = var_new.where(mask, var_old)
            dataset[var_name].encoding.update(fill_value=var_fill_value)

    return dataset


def mask_dataset_by_geometry(dataset: xr.Dataset,
                             geometry: GeometryLike,
                             excluded_vars: Sequence[str] = None,
                             no_clip: bool = False,
                             save_geometry_mask: Union[str, bool] = False,
                             save_geometry_wkt: Union[str, bool] = False) -> Optional[xr.Dataset]:
    """
    Mask a dataset according to the given geometry. The cells of variables of the
    returned dataset will have NaN-values where their spatial coordinates are not intersecting
    the given geometry.

    :param dataset: The dataset
    :param geometry: A geometry-like object, see py:function:`convert_geometry`.
    :param excluded_vars: Optional sequence of names of data variables that should not be masked
        (but still may be clipped).
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
    xy_var_names = get_dataset_xy_var_names(dataset, must_exist=True)
    intersection_geometry = intersect_geometries(get_dataset_bounds(dataset, xy_var_names=xy_var_names),
                                                 geometry)
    if intersection_geometry is None:
        return None

    if not no_clip:
        dataset = _clip_dataset_by_geometry(dataset, intersection_geometry, xy_var_names)

    ds_x_min, ds_y_min, ds_x_max, ds_y_max = get_dataset_bounds(dataset, xy_var_names=xy_var_names)

    x_var_name, y_var_name = xy_var_names
    x_var, y_var = dataset[x_var_name], dataset[y_var_name]

    width = x_var.size
    height = y_var.size
    spatial_res = (ds_x_max - ds_x_min) / width

    mask_data = get_geometry_mask(width, height, intersection_geometry, ds_x_min, ds_y_min, spatial_res)
    mask = xr.DataArray(mask_data,
                        coords={y_var_name: y_var, x_var_name: x_var},
                        dims=(y_var.dims[0], x_var.dims[0]))

    dataset_vars = {}
    for var_name, var in dataset.data_vars.items():
        if not excluded_vars or var_name not in excluded_vars:
            dataset_vars[var_name] = var.where(mask)
        else:
            dataset_vars[var_name] = var

    masked_dataset = xr.Dataset(dataset_vars, coords=dataset.coords, attrs=dataset.attrs)

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
    xy_var_names = get_dataset_xy_var_names(dataset, must_exist=True)
    intersection_geometry = intersect_geometries(get_dataset_bounds(dataset, xy_var_names=xy_var_names), geometry)
    if intersection_geometry is None:
        return None
    return _clip_dataset_by_geometry(dataset, intersection_geometry, xy_var_names, save_geometry_wkt=save_geometry_wkt)


def _clip_dataset_by_geometry(dataset: xr.Dataset,
                              intersection_geometry: shapely.geometry.base.BaseGeometry,
                              xy_var_names: Tuple[str, str],
                              save_geometry_wkt: bool = False) -> Optional[xr.Dataset]:
    # TODO (forman): the following code is wrong, if the dataset bounds cross the anti-meridian!

    ds_x_min, ds_y_min, ds_x_max, ds_y_max = get_dataset_bounds(dataset, xy_var_names=xy_var_names)

    x_var_name, y_var_name = xy_var_names
    x_var = dataset[x_var_name]
    y_var = dataset[y_var_name]

    width = x_var.size
    height = y_var.size
    res = (ds_y_max - ds_y_min) / height

    g_x_min, g_y_min, g_x_max, g_y_max = intersection_geometry.bounds
    x1 = _clamp(int(math.floor((g_x_min - ds_x_min) / res)), 0, width - 1)
    x2 = _clamp(int(math.ceil((g_x_max - ds_x_min) / res)), 0, width - 1)
    y1 = _clamp(int(math.floor((g_y_min - ds_y_min) / res)), 0, height - 1)
    y2 = _clamp(int(math.ceil((g_y_max - ds_y_min) / res)), 0, height - 1)
    if not is_dataset_y_axis_inverted(dataset, xy_var_names=xy_var_names):
        _y1, _y2 = y1, y2
        y1 = height - _y2 - 1
        y2 = height - _y1 - 1

    dataset_subset = dataset.isel(**{x_var_name: slice(x1, x2), y_var_name: slice(y1, y2)})

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
                      x_min: float, y_min: float, res: float) -> np.ndarray:
    geometry = convert_geometry(geometry)
    # noinspection PyTypeChecker
    transform = affine.Affine(res, 0.0, x_min,
                              0.0, -res, y_min + res * height)
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


def is_dataset_y_axis_inverted(dataset: Union[xr.Dataset, xr.DataArray],
                               xy_var_names: Tuple[str, str] = None) -> bool:
    if xy_var_names is None:
        xy_var_names = get_dataset_xy_var_names(dataset, must_exist=True)
    y_var = dataset[xy_var_names[1]]
    return float(y_var[0]) < float(y_var[-1])


def is_lon_lat_dataset(dataset: Union[xr.Dataset, xr.DataArray],
                       xy_var_names: Tuple[str, str] = None) -> bool:
    if xy_var_names is None:
        xy_var_names = get_dataset_xy_var_names(dataset, must_exist=True)
    x_var_name, y_var_name = xy_var_names
    if x_var_name == 'lon' and y_var_name == 'lat':
        return True
    x_var = dataset[x_var_name]
    y_var = dataset[y_var_name]
    return x_var.attrs.get('long_name') == 'longitude' and y_var.attrs.get('long_name') == 'latitude'


def get_dataset_geometry(dataset: Union[xr.Dataset, xr.DataArray],
                         xy_var_names: Tuple[str, str] = None) -> shapely.geometry.base.BaseGeometry:
    if xy_var_names is None:
        xy_var_names = get_dataset_xy_var_names(dataset, must_exist=True)
    geo_bounds = get_dataset_bounds(dataset, xy_var_names=xy_var_names)
    if is_lon_lat_dataset(dataset, xy_var_names=xy_var_names):
        return get_box_split_bounds_geometry(*geo_bounds)
    else:
        return shapely.geometry.box(*geo_bounds)


def get_dataset_bounds(dataset: Union[xr.Dataset, xr.DataArray],
                       xy_var_names: Tuple[str, str] = None) -> Bounds:
    if xy_var_names is None:
        xy_var_names = get_dataset_xy_var_names(dataset, must_exist=True)
    x_name, y_name = xy_var_names
    x_var, y_var = dataset.coords[x_name], dataset.coords[y_name]

    x_bnds_name = get_dataset_bounds_var_name(dataset, x_name)
    if x_bnds_name:
        x_bnds_var = dataset.coords[x_bnds_name]
        x_min = x_bnds_var[0][0]
        x_max = x_bnds_var[-1][1]
    else:
        x_min = x_var[0]
        x_max = x_var[-1]
        delta = min(abs(np.diff(x_var)))
        x_min -= 0.5 * delta
        x_max += 0.5 * delta

    y_bnds_name = get_dataset_bounds_var_name(dataset, y_name)
    if y_bnds_name:
        y_bnds_var = dataset.coords[y_bnds_name]
        y1 = y_bnds_var[0][0]
        y2 = y_bnds_var[-1][1]
        y_min = min(y1, y2)
        y_max = max(y1, y2)
    else:
        y1 = y_var[0]
        y2 = y_var[-1]
        delta = min(abs(np.diff(y_var)))
        y_min = min(y1, y2) - 0.5 * delta
        y_max = max(y1, y2) + 0.5 * delta

    return float(x_min), float(y_min), float(x_max), float(y_max)


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
