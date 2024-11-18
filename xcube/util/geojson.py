# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Any, Optional, Dict
from collections.abc import Sequence

from xcube.util.undefined import UNDEFINED


class GeoJSON:
    PRIMITIVE_GEOMETRY_TYPES = {
        "Point",
        "LineString",
        "Polygon",
        "MultiPoint",
        "MultiLineString",
        "MultiPolygon",
    }
    GEOMETRY_COLLECTION_TYPE = "GeometryCollection"
    FEATURE_TYPE = "Feature"
    FEATURE_COLLECTION_TYPE = "FeatureCollection"

    @classmethod
    def is_point(cls, obj: Any) -> bool:
        return cls.get_type_name(obj) == "Point"

    @classmethod
    def is_feature(cls, obj: Any) -> bool:
        return cls.get_type_name(obj) == cls.FEATURE_TYPE

    @classmethod
    def is_feature_collection(cls, obj: Any) -> bool:
        return cls.get_type_name(
            obj
        ) == cls.FEATURE_COLLECTION_TYPE and cls._is_valid_sequence(obj, "features")

    @classmethod
    def is_geometry(cls, obj: Any) -> bool:
        type_name = cls.get_type_name(obj)
        if type_name in cls.PRIMITIVE_GEOMETRY_TYPES:
            return cls._is_valid_sequence(obj, "coordinates")
        if type_name == cls.GEOMETRY_COLLECTION_TYPE:
            return cls._is_valid_sequence(obj, "geometries")
        return False

    @classmethod
    def is_geometry_collection(cls, obj: Any) -> bool:
        type_name = cls.get_type_name(obj)
        if type_name == cls.GEOMETRY_COLLECTION_TYPE:
            return cls._is_valid_sequence(obj, "geometries")
        return False

    @classmethod
    def get_geometry_collection_geometries(cls, obj: Any) -> Optional[Sequence[dict]]:
        type_name = cls.get_type_name(obj)
        if type_name == cls.GEOMETRY_COLLECTION_TYPE:
            geometries = cls._get_sequence(obj, "geometries")
            if geometries == UNDEFINED:
                return None
            elif geometries is None:
                return []
            else:
                return geometries
        return None

    @classmethod
    def get_feature_collection_features(cls, obj: Any) -> Optional[Sequence[dict]]:
        type_name = cls.get_type_name(obj)
        if type_name == cls.FEATURE_COLLECTION_TYPE:
            features = cls._get_sequence(obj, "features")
            if features == UNDEFINED:
                return None
            elif features is None:
                return []
            else:
                return features
        return None

    @classmethod
    def get_feature_geometry(cls, obj: Any) -> Optional[dict]:
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
        return obj["type"] or None

    @classmethod
    def _is_valid_sequence(cls, obj: Any, attr_name: str) -> bool:
        return cls._get_sequence(obj, attr_name) != UNDEFINED

    @classmethod
    def _get_sequence(cls, obj: Any, attr_name: str) -> Optional[Sequence[Any]]:
        if attr_name not in obj:
            return UNDEFINED
        sequence = obj[attr_name]
        if sequence is None:
            # GeoJSON sequence properties, 'coordinates', 'geometries', 'features', may be None according to spec
            return None
        try:
            iter(sequence)
            return sequence
        except TypeError:
            return UNDEFINED
