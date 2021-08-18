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

from typing import Any, Optional, Dict, Sequence

from xcube.util.undefined import UNDEFINED


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
    def is_feature(cls, obj: Any) -> bool:
        return cls.get_type_name(obj) == cls.FEATURE_TYPE

    @classmethod
    def is_feature_collection(cls, obj: Any) -> bool:
        return cls.get_type_name(obj) == cls.FEATURE_COLLECTION_TYPE and cls._is_valid_sequence(obj, 'features')

    @classmethod
    def is_geometry(cls, obj: Any) -> bool:
        type_name = cls.get_type_name(obj)
        if type_name in cls.PRIMITIVE_GEOMETRY_TYPES:
            return cls._is_valid_sequence(obj, 'coordinates')
        if type_name == cls.GEOMETRY_COLLECTION_TYPE:
            return cls._is_valid_sequence(obj, 'geometries')
        return False

    @classmethod
    def is_geometry_collection(cls, obj: Any) -> bool:
        type_name = cls.get_type_name(obj)
        if type_name == cls.GEOMETRY_COLLECTION_TYPE:
            return cls._is_valid_sequence(obj, 'geometries')
        return False

    @classmethod
    def get_geometry_collection_geometries(cls, obj: Any) -> Optional[Sequence[Dict]]:
        type_name = cls.get_type_name(obj)
        if type_name == cls.GEOMETRY_COLLECTION_TYPE:
            geometries = cls._get_sequence(obj, 'geometries')
            if geometries == UNDEFINED:
                return None
            elif geometries is None:
                return []
            else:
                return geometries
        return None

    @classmethod
    def get_feature_collection_features(cls, obj: Any) -> Optional[Sequence[Dict]]:
        type_name = cls.get_type_name(obj)
        if type_name == cls.FEATURE_COLLECTION_TYPE:
            features = cls._get_sequence(obj, 'features')
            if features == UNDEFINED:
                return None
            elif features is None:
                return []
            else:
                return features
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
