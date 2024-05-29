# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Any, Dict, List, Optional, Callable
from collections.abc import Iterator
from collections.abc import Sequence

import fiona
import fsspec
import pyproj
import shapely

from fiona.collection import Collection

from xcube.server.api import ApiError
from xcube.server.api import Context
from xcube.webapi.common.context import ResourcesContext

PlaceGroup = dict[str, Any]
Feature = dict[str, Any]

ALL_PLACES = "all"


class PlacesContext(ResourcesContext):
    _feature_index: int = 0

    def __init__(self, server_ctx: Context):
        super().__init__(server_ctx)
        self._additional_place_groups: dict[str, list[PlaceGroup]] = dict()
        self._place_group_cache: dict[str, PlaceGroup] = dict()

    def on_dispose(self):
        if self._place_group_cache:
            self._place_group_cache.clear()

    def get_cached_place_group(self, place_group_id: str) -> Optional[PlaceGroup]:
        return self._place_group_cache.get(place_group_id)

    def set_cached_place_group(self, place_group_id: str, place_group: PlaceGroup):
        self._place_group_cache[place_group_id] = place_group

    def get_cached_place_groups(
        self, predicate: Optional[Callable[[str, PlaceGroup], bool]] = None
    ) -> list[PlaceGroup]:
        return [
            v
            for k, v in self._place_group_cache.items()
            if predicate is None or predicate(k, v)
        ]

    def new_feature_id(self) -> str:
        with self.rlock:
            feature_id = str(self._feature_index)
            self._feature_index += 1
            return feature_id

    def get_global_place_groups(
        self, base_url: str, load_features=False
    ) -> list[PlaceGroup]:
        return self.load_place_groups(
            self.config.get("PlaceGroups", []),
            base_url,
            is_global=True,
            load_features=load_features,
        )

    def get_global_place_group(
        self, place_group_id: str, base_url: str, load_features: bool = False
    ) -> PlaceGroup:
        place_group_config = self._get_place_group_config(place_group_id)
        return self._load_place_group(
            place_group_config, base_url, is_global=True, load_features=load_features
        )

    def _get_place_group_config(self, place_group_id: str) -> dict:
        place_group_configs = self.config.get("PlaceGroups", [])
        for place_group_config in place_group_configs:
            if place_group_config["Identifier"] == place_group_id:
                return place_group_config
        raise ApiError.NotFound(f'Place group "{place_group_id}" not found')

    def load_place_groups(
        self,
        place_group_configs: list,
        base_url: str,
        is_global: bool = False,
        load_features: bool = False,
        qualifiers: list[str] = list(),
    ) -> list[PlaceGroup]:
        place_groups = []
        for place_group_config in place_group_configs:
            place_group = self._load_place_group(
                place_group_config,
                base_url,
                is_global=is_global,
                load_features=load_features,
            )
            place_groups.append(place_group)
        for q in [
            qualifier
            for qualifier in qualifiers
            if qualifier in self._additional_place_groups
        ]:
            for place_group in self._additional_place_groups[q]:
                place_groups.append(place_group)
        return place_groups

    def add_place_group(self, place_group: PlaceGroup, qualifiers: list[str] = list()):
        for qualifier in qualifiers:
            if qualifier not in self._additional_place_groups:
                self._additional_place_groups[qualifier] = []
            self._additional_place_groups[qualifier].append(place_group)

    def _load_place_group(
        self,
        place_group_config: dict[str, Any],
        base_url: str,
        is_global: bool = False,
        load_features: bool = False,
    ) -> PlaceGroup:
        place_group_id = place_group_config.get("PlaceGroupRef")
        if place_group_id:
            if is_global:
                raise ApiError.InvalidServerConfig(
                    "'PlaceGroupRef' cannot be used in a global place group"
                )
            if len(place_group_config) > 1:
                raise ApiError.InvalidServerConfig(
                    "'PlaceGroupRef' if present,"
                    " must be the only entry in a 'PlaceGroups' item"
                )
            return self.get_global_place_group(
                place_group_id, base_url, load_features=load_features
            )

        place_group_id = self.get_place_group_id_safe(place_group_config)

        place_group = self.get_cached_place_group(place_group_id)
        if place_group is None:
            place_group_title = place_group_config.get("Title", place_group_id)
            place_path_wc = self.get_config_path(
                place_group_config, f"'PlaceGroups' item"
            )
            fs, place_path = fsspec.core.url_to_fs(place_path_wc)
            source_paths = [fs.unstrip_protocol(p) for p in fs.glob(place_path)]
            source_encoding = place_group_config.get("CharacterEncoding", "utf-8")

            join = None
            place_join = place_group_config.get("Join")
            if isinstance(place_join, dict):
                join_path = self.get_config_path(
                    place_join, "'Join' of a 'PlaceGroups' item"
                )
                join_property = place_join.get("Property")
                if not join_property:
                    raise ApiError.InvalidServerConfig(
                        "Missing 'Property' entry in 'Join'" " of a 'PlaceGroups' item"
                    )
                join_encoding = place_join.get("CharacterEncoding", "utf-8")
                join = dict(
                    path=join_path, property=join_property, encoding=join_encoding
                )

            property_mapping = self.get_property_mapping(base_url, place_group_config)

            place_group = dict(
                type="FeatureCollection",
                features=None,
                id=place_group_id,
                title=place_group_title,
                propertyMapping=property_mapping,
                sourcePaths=source_paths,
                sourceEncoding=source_encoding,
                join=join,
            )

            self.check_sub_group_configs(place_group_config)
            self.set_cached_place_group(place_group_id, place_group)

        if load_features:
            self.load_place_group_features(place_group)

        return place_group

    @staticmethod
    def get_place_group_id_safe(place_group_config):
        place_group_id = place_group_config.get("Identifier")
        if not place_group_id:
            raise ApiError.InvalidServerConfig(
                "Missing 'Identifier' entry in a 'PlaceGroups' item"
            )
        return place_group_id

    @staticmethod
    def check_sub_group_configs(place_group_config):
        sub_place_group_configs = place_group_config.get("Places")
        if sub_place_group_configs:
            raise ApiError.InvalidServerConfig(
                "Invalid 'Places' entry in a 'PlaceGroups' item:" " not implemented yet"
            )

    @staticmethod
    def get_property_mapping(base_url, place_group_config):
        property_mapping = place_group_config.get("PropertyMapping")
        if property_mapping:
            property_mapping = dict(property_mapping)
            for key, value in property_mapping.items():
                if isinstance(value, str) and "${base_url}" in value:
                    property_mapping[key] = value.replace("${base_url}", base_url)
        return property_mapping

    def load_place_group_features(
        self, place_group: PlaceGroup
    ) -> list[dict[str, Any]]:
        features = place_group.get("features")
        if features is not None:
            return features
        source_paths = place_group["sourcePaths"]
        source_encoding = place_group["sourceEncoding"]
        features = []
        for source_path in source_paths:
            with self.measure_time(f"Loading features from file {source_path}"):
                with fiona.open(
                    source_path, encoding=source_encoding
                ) as feature_collection:
                    for feature in self._to_geo_interface(feature_collection):
                        self._remove_feature_id(feature)
                        feature["id"] = self.new_feature_id()
                        features.append(feature)

        join = place_group["join"]
        if join:
            join_path = join["path"]
            join_property = join["property"]
            join_encoding = join["encoding"]
            with fiona.open(join_path, encoding=join_encoding) as feature_collection:
                indexed_join_features = self._get_indexed_features(
                    list(self._to_geo_interface(feature_collection)), join_property
                )
            for feature in features:
                properties = feature.get("properties")
                if isinstance(properties, dict) and join_property in properties:
                    join_value = properties[join_property]
                    join_feature = indexed_join_features.get(join_value)
                    if join_feature:
                        join_properties = join_feature.get("properties")
                        if join_properties:
                            properties.update(join_properties)
                            feature["properties"] = properties

        place_group["features"] = features
        return features

    @classmethod
    def _to_geo_interface(
        cls, feature_collection: Collection
    ) -> Iterator[dict[str, Any]]:
        source_crs = feature_collection.crs
        target_crs = fiona.crs.CRS.from_epsg(4326)
        if not source_crs == target_crs:
            project = pyproj.Transformer.from_crs(
                source_crs, target_crs, always_xy=True
            ).transform
        for feature in feature_collection:
            # fiona >=1.9 returns features of type fiona.model.Feature
            # rather than JSON-serializable dictionaries.
            if hasattr(feature, "__geo_interface__"):
                # We fall back on the traditional geo-interface:
                feature = feature.__geo_interface__
                # Fiona =1.9.0 adds empty "geometries" field
                # to any "geometry", we fix this too:
                geometry = feature.get("geometry")
                if (
                    geometry
                    and "geometries" in geometry
                    and geometry.get("type") != "GeometryCollection"
                ):
                    del geometry["geometries"]
            if not source_crs == target_crs:
                geometry = feature.get("geometry")
                shapely_geom = shapely.geometry.shape(geometry)
                feature["geometry"] = shapely.ops.transform(
                    project, shapely_geom
                ).__geo_interface__
            yield feature

    @classmethod
    def _get_indexed_features(
        cls, features: Sequence[Feature], property_name: str
    ) -> Feature:
        feature_index = {}
        for feature in features:
            properties = feature.get("properties")
            if properties and property_name in properties:
                property_value = properties[property_name]
                feature_index[property_value] = feature
        return feature_index

    @classmethod
    def _remove_feature_id(cls, feature: Feature):
        if "id" in feature:
            del feature["id"]
        if "ID" in feature:
            del feature["ID"]
