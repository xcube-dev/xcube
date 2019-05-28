from typing import Any, Optional, List, Dict


class GeoJSON:
    PRIMITIVE_GEOMETRY_TYPES = {'Point', 'LineString', 'Polygon',
                                'MultiPoint', 'MultiLineString', 'MultiPolygon'}
    GEOMETRY_COLLECTION_TYPE = 'GeometryCollection'
    FEATURE_TYPE = 'Feature'
    FEATURE_COLLECTION_TYPE = 'FeatureCollection'

    @classmethod
    def is_point(cls, obj: Any) -> bool:
        return cls.get_type_name(obj) == 'Point'

    @classmethod
    def is_geometry(cls, obj: Any) -> bool:
        type_name = cls.get_type_name(obj)
        if type_name in cls.PRIMITIVE_GEOMETRY_TYPES:
            if 'coordinates' not in obj:
                return False
            coordinates = obj['coordinates']
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
            if 'geometries' not in obj:
                return None
            geometries = obj['geometries']
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
            features = obj['features']
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
            if 'geometry' not in obj:
                return None
            geometry = obj['geometry']
            if cls.is_geometry(geometry):
                return geometry
            return None
        return None

    @classmethod
    def get_type_name(cls, obj: Any) -> Optional[str]:
        if not isinstance(obj, dict) or "type" not in obj:
            return None
        return obj["type"] or None
