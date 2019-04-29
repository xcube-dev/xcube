from typing import Optional, Tuple, Union, Dict, Any, List

import affine
import numpy as np
import pandas as pd
import rasterio.features
import shapely.geometry
import shapely.geometry
import xarray as xr

Bounds = Tuple[float, float, float, float]
SplitBounds = Tuple[Bounds, Optional[Bounds]]


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


def get_geometry_mask(width: int, height: int,
                      geometry: Union[shapely.geometry.base.BaseGeometry, Dict],
                      lon_min: float, lat_min: float, res: float) -> np.ndarray:
    # noinspection PyTypeChecker
    transform = affine.Affine(res, 0.0, lon_min,
                              0.0, -res, lat_min + res * height)
    return rasterio.features.geometry_mask([geometry],
                                           out_shape=(height, width),
                                           transform=transform,
                                           all_touched=True,
                                           invert=True)


def timestamp_to_iso_string(time: np.datetime64, freq='S'):
    """
    Convert a UTC timestamp given as nanos, millis, seconds, etc. since 1970-01-01 00:00:00
    to an ISO-format string.

    :param time: UTC timestamp given as time delta since since 1970-01-01 00:00:00 in the units given by
           the numpy datetime64 type, so it can be as nanos, millis, seconds since 1970-01-01 00:00:00.
    :param freq: time rounding resolution. See pandas.Timestamp.round().
    :return: ISO-format string.
    """
    # All times are UTC (Z = Zulu Time Zone = UTC)
    return pd.Timestamp(time).round(freq).isoformat() + 'Z'


class GeoJSON:
    PRIMITIVE_GEOMETRY_TYPES = {"Point", "LineString", "Polygon",
                                "MultiPoint", "MultiLineString", "MultiPolygon"}
    GEOMETRY_COLLECTION_TYPE = "GeometryCollection"
    FEATURE_TYPE = "Feature"
    FEATURE_COLLECTION_TYPE = "FeatureCollection"

    @classmethod
    def is_geometry(cls, obj: Any) -> bool:
        type_name = cls.get_type_name(obj)
        if type_name in cls.PRIMITIVE_GEOMETRY_TYPES:
            if "coordinates" not in obj:
                return False
            coordinates = obj["coordinates"]
            return coordinates is None or isinstance(coordinates, list)
        if type_name == cls.GEOMETRY_COLLECTION_TYPE:
            if "geometries" not in obj:
                return False
            geometries = obj["geometries"]
            return geometries is None or isinstance(geometries, list)
        return False

    @classmethod
    def get_geometry_collection_geometries(cls, obj: Any) -> Optional[List[Dict]]:
        type_name = cls.get_type_name(obj)
        if type_name == cls.GEOMETRY_COLLECTION_TYPE:
            if "geometries" not in obj:
                return None
            geometries = obj["geometries"]
            if geometries is None:
                return []
            if isinstance(geometries, list):
                return geometries
            return None
        return None

    @classmethod
    def get_feature_collection_features(cls, obj: Any) -> Optional[List[Dict]]:
        type_name = cls.get_type_name(obj)
        if type_name == cls.FEATURE_COLLECTION_TYPE:
            if "features" not in obj:
                return None
            features = obj["features"]
            if features is None:
                return []
            if isinstance(features, list):
                return features
            return None
        return None

    @classmethod
    def get_feature_geometry(cls, obj: Any) -> Optional[Dict]:
        type_name = cls.get_type_name(obj)
        if type_name == cls.FEATURE_TYPE:
            if "geometry" not in obj:
                return None
            geometry = obj["geometry"]
            if cls.is_geometry(geometry):
                return geometry
            return None
        return None

    @classmethod
    def get_type_name(cls, obj: Any) -> Optional[str]:
        if not isinstance(obj, dict) or "type" not in obj:
            return None
        if "type" not in obj:
            return None
        return obj["type"] or None
