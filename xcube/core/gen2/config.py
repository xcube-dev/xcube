# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import collections.abc
import numbers
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional, Tuple, Union

import pyproj

from xcube.util.assertions import assert_given, assert_instance, assert_true
from xcube.util.jsonschema import (
    JsonArraySchema,
    JsonBooleanSchema,
    JsonDateSchema,
    JsonIntegerSchema,
    JsonNumberSchema,
    JsonObject,
    JsonObjectSchema,
    JsonStringSchema,
)


class InputConfig(JsonObject):
    def __init__(
        self,
        store_id: str = None,
        opener_id: str = None,
        data_id: str = None,
        store_params: Mapping[str, Any] = None,
        open_params: Mapping[str, Any] = None,
    ):
        assert_true(
            store_id or opener_id, "One of store_id and opener_id must be given"
        )
        assert_given(data_id, "data_id")
        self.store_id = store_id
        self.opener_id = opener_id
        self.data_id = data_id
        self.store_params = store_params
        self.open_params = open_params

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                store_id=JsonStringSchema(min_length=1),
                opener_id=JsonStringSchema(min_length=1),
                data_id=JsonStringSchema(min_length=1),
                store_params=JsonObjectSchema(
                    additional_properties=True, nullable=True
                ),
                open_params=JsonObjectSchema(additional_properties=True, nullable=True),
            ),
            additional_properties=False,
            required=["data_id"],
            factory=cls,
        )


class CallbackConfig(JsonObject):
    def __init__(self, api_uri: str = None, access_token: str = None):
        assert_true(
            api_uri and access_token, "Both, api_uri and access_token must be given"
        )
        self.api_uri = api_uri
        self.access_token = access_token

    @classmethod
    def get_schema(cls):
        return JsonObjectSchema(
            properties=dict(
                api_uri=JsonStringSchema(min_length=1),
                access_token=JsonStringSchema(min_length=1),
            ),
            additional_properties=False,
            required=["api_uri", "access_token"],
            factory=cls,
        )


class OutputConfig(JsonObject):
    def __init__(
        self,
        store_id: str = None,
        writer_id: str = None,
        data_id: str = None,
        store_params: Mapping[str, Any] = None,
        write_params: Mapping[str, Any] = None,
        replace: bool = None,
    ):
        assert_true(
            store_id or writer_id, "One of store_id and writer_id must be given"
        )
        self.store_id = store_id
        self.writer_id = writer_id
        self.data_id = data_id
        self.store_params = store_params
        self.write_params = write_params
        self.replace = replace

    @classmethod
    def get_schema(cls):
        return JsonObjectSchema(
            properties=dict(
                store_id=JsonStringSchema(min_length=1),
                writer_id=JsonStringSchema(min_length=1),
                data_id=JsonStringSchema(default=None),
                store_params=JsonObjectSchema(
                    additional_properties=True, nullable=True
                ),
                write_params=JsonObjectSchema(
                    additional_properties=True, nullable=True
                ),
                replace=JsonBooleanSchema(default=False),
            ),
            additional_properties=False,
            required=[],
            factory=cls,
        )


# Need to be aligned with params in resample_cube(cube, **params)
class CubeConfig(JsonObject):
    def __init__(
        self,
        variable_names: Sequence[str] = None,
        crs: str = None,
        bbox: tuple[float, float, float, float] = None,
        spatial_res: Union[float, tuple[float]] = None,
        tile_size: Union[int, tuple[int, int]] = None,
        time_range: tuple[str, Optional[str]] = None,
        time_period: str = None,
        chunks: Mapping[str, Optional[int]] = None,
        metadata: Mapping[str, Any] = None,
        variable_metadata: Mapping[str, Mapping[str, Any]] = None,
    ):
        self.variable_names = None
        if variable_names is not None:
            assert_true(len(variable_names) > 0, "variable_names is invalid")
            self.variable_names = tuple(map(str, variable_names))

        self.crs = None
        if crs is not None:
            assert_instance(crs, str, "crs")
            try:
                pyproj.crs.CRS.from_string(crs)
            except pyproj.exceptions.CRSError:
                raise ValueError("crs is invalid")
            self.crs = crs

        self.bbox = None
        if bbox is not None:
            assert_true(len(bbox) == 4, "bbox is invalid")
            self.bbox = tuple(map(float, bbox))

        self.spatial_res = None
        if spatial_res is not None:
            assert_instance(spatial_res, numbers.Number, "spatial_res")
            self.spatial_res = float(spatial_res)

        self.tile_size = None
        if tile_size is not None:
            if isinstance(tile_size, int):
                tile_width, tile_height = tile_size, tile_size
            else:
                try:
                    tile_width, tile_height = tile_size
                except (ValueError, TypeError):
                    raise ValueError(
                        "tile_size must be an integer or a pair of integers"
                    )
                assert_instance(tile_width, numbers.Number, "tile_width of tile_size")
                assert_instance(tile_height, numbers.Number, "tile_height of tile_size")
            self.tile_size = tile_width, tile_height

        self.time_range = None
        if time_range is not None:
            assert_true(len(time_range) == 2, "time_range is invalid")
            self.time_range = tuple(time_range)

        self.time_period = None
        if time_period is not None:
            assert_instance(time_period, str, "time_period")
            self.time_period = time_period

        self.chunks = None
        if chunks is not None:
            assert_instance(chunks, collections.abc.Mapping, "chunks")
            for chunk_dim, chunk_size in chunks.items():
                assert_instance(chunk_dim, str, "chunk dimension")
                assert_instance(chunk_size, (int, type(None)), "chunk size")
            self.chunks = dict(chunks)

        self.metadata = None
        if metadata is not None:
            assert_instance(metadata, collections.abc.Mapping, "metadata")
            self.metadata = dict(metadata)

        self.variable_metadata = None
        if variable_metadata is not None:
            assert_instance(
                variable_metadata, collections.abc.Mapping, "variable_metadata"
            )
            for var_name, metadata in variable_metadata.items():
                assert_instance(var_name, str, "chunk dimension")
                assert_instance(metadata, collections.abc.Mapping, "variable metadata")
            self.variable_metadata = dict(variable_metadata)

    def drop_props(self, name: Union[str, Iterable[str]]):
        """Drop one or more named properties from this configuration.

        Args:
            name: the name of the property to be dropped.
            names: more names of properties to be dropped.

        Returns:
            a new cube configuration.
        """
        if not name:
            return self
        name_set = {name} if isinstance(name, str) else set(name)
        for k in name_set:
            assert_true(
                hasattr(self, k), message=f"{k} is not a property of {CubeConfig!r}"
            )
        return self.from_dict(
            {k: v for k, v in self.to_dict().items() if k not in name_set}
        )

    @classmethod
    def get_schema(cls):
        return JsonObjectSchema(
            properties=dict(
                variable_names=JsonArraySchema(
                    nullable=True, items=JsonStringSchema(min_length=1), min_items=0
                ),
                crs=JsonStringSchema(nullable=True, min_length=1),
                bbox=JsonArraySchema(
                    nullable=True,
                    items=[
                        JsonNumberSchema(),
                        JsonNumberSchema(),
                        JsonNumberSchema(),
                        JsonNumberSchema(),
                    ],
                ),
                spatial_res=JsonNumberSchema(nullable=True, exclusive_minimum=0.0),
                tile_size=JsonArraySchema(
                    nullable=True,
                    items=[
                        JsonIntegerSchema(minimum=1, maximum=2500),
                        JsonIntegerSchema(minimum=1, maximum=2500),
                    ],
                ),
                time_range=JsonDateSchema.new_range(nullable=True),
                time_period=JsonStringSchema(
                    nullable=True, pattern=r"^([1-9][0-9]*)?[DWMY]$"
                ),
                chunks=JsonObjectSchema(
                    nullable=True,
                    additional_properties=JsonIntegerSchema(nullable=True, minimum=1),
                ),
                metadata=JsonObjectSchema(nullable=True, additional_properties=True),
                variable_metadata=JsonObjectSchema(
                    nullable=True,
                    additional_properties=JsonObjectSchema(additional_properties=True),
                ),
            ),
            additional_properties=False,
            factory=cls,
        )
