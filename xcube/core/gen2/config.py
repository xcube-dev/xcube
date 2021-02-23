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

import numbers
from collections import Iterable
from typing import Optional, Dict, Any, Sequence, Mapping, Tuple, Union

import pyproj

from xcube.util.assertions import assert_condition
from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonDateSchema
from xcube.util.jsonschema import JsonIntegerSchema
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema


class InputConfig:
    def __init__(self,
                 store_id: str = None,
                 opener_id: str = None,
                 data_id: str = None,
                 store_params: Mapping[str, Any] = None,
                 open_params: Mapping[str, Any] = None):
        assert_condition(store_id or opener_id, 'One of store_id and opener_id must be given')
        assert_given(data_id, 'data_id')
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
                store_params=JsonObjectSchema(additional_properties=True, nullable=True),
                open_params=JsonObjectSchema(additional_properties=True, nullable=True)
            ),
            additional_properties=False,
            required=['data_id'],
            factory=cls,
        )

    def to_dict(self):
        return _to_dict(self, ('store_id',
                               'opener_id',
                               'data_id',
                               'store_params',
                               'open_params'))


class CallbackConfig:
    def __init__(self,
                 api_uri: str = None,
                 access_token: str = None):
        assert_condition(api_uri and access_token, 'Both, api_uri and access_token must be given')
        self.api_uri = api_uri
        self.access_token = access_token

    @classmethod
    def get_schema(cls):
        return JsonObjectSchema(
            properties=dict(
                api_uri=JsonStringSchema(min_length=1),
                access_token=JsonStringSchema(min_length=1)
            ),
            additional_properties=False,
            required=["api_uri", "access_token"],
            factory=cls,
        )

    def to_dict(self) -> dict:
        return _to_dict(self, ('api_uri', 'access_token'))


class OutputConfig:

    def __init__(self,
                 store_id: str = None,
                 writer_id: str = None,
                 data_id: str = None,
                 store_params: Mapping[str, Any] = None,
                 write_params: Mapping[str, Any] = None,
                 replace: bool = None):
        assert_condition(store_id or writer_id, 'One of store_id and writer_id must be given')
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
                store_params=JsonObjectSchema(additional_properties=True, nullable=True),
                write_params=JsonObjectSchema(additional_properties=True, nullable=True),
                replace=JsonBooleanSchema(default=False),
            ),
            additional_properties=False,
            required=[],
            factory=cls,
        )

    def to_dict(self):
        return _to_dict(self, ('store_id',
                               'writer_id',
                               'data_id',
                               'store_params',
                               'write_params',
                               'replace'))


# Need to be aligned with params in resample_cube(cube, **params)
class CubeConfig:

    def __init__(self,
                 variable_names: Sequence[str] = None,
                 crs: str = None,
                 bbox: Tuple[float, float, float, float] = None,
                 spatial_res: Union[float, Tuple[float]] = None,
                 tile_size: Union[int, Tuple[int, int]] = None,
                 time_range: Tuple[str, Optional[str]] = None,
                 time_period: str = None):

        self.variable_names = None
        if variable_names is not None:
            assert_condition(len(variable_names) > 0, 'variable_names is invalid')
            self.variable_names = tuple(map(str, variable_names))

        self.crs = None
        if crs is not None:
            assert_instance(crs, str, 'crs')
            try:
                pyproj.crs.CRS.from_string(crs)
            except pyproj.exceptions.CRSError:
                raise ValueError('crs is invalid')
            self.crs = crs

        self.bbox = None
        if bbox is not None:
            assert_condition(len(bbox) == 4, 'bbox is invalid')
            self.bbox = tuple(map(float, bbox))

        self.spatial_res = None
        if spatial_res is not None:
            assert_instance(spatial_res, numbers.Number, 'spatial_res')
            self.spatial_res = float(spatial_res)

        self.tile_size = None
        if tile_size is not None:
            if isinstance(tile_size, int):
                tile_width, tile_height = tile_size, tile_size
            else:
                try:
                    tile_width, tile_height = tile_size
                except (ValueError, TypeError):
                    raise ValueError('tile_size must be an integer or a pair of integers')
                assert_instance(tile_width, numbers.Number, 'tile_width of tile_size')
                assert_instance(tile_height, numbers.Number, 'tile_height of tile_size')
            self.tile_size = tile_width, tile_height

        self.time_range = None
        if time_range is not None:
            assert_condition(len(time_range) == 2, 'time_range is invalid')
            self.time_range = tuple(time_range)

        self.time_period = None
        if time_period is not None:
            assert_instance(time_period, str, 'time_period')
            self.time_period = time_period

    def to_dict(self):
        return _to_dict(self, ('variable_names',
                               'crs',
                               'bbox',
                               'spatial_res',
                               'tile_size',
                               'time_range',
                               'time_period'))

    @classmethod
    def get_schema(cls):
        return JsonObjectSchema(
            properties=dict(
                variable_names=JsonArraySchema(
                    nullable=True,
                    items=JsonStringSchema(min_length=1),
                    min_items=0
                ),
                crs=JsonStringSchema(
                    nullable=True,
                    min_length=1
                ),
                bbox=JsonArraySchema(
                    nullable=True,
                    items=[JsonNumberSchema(),
                           JsonNumberSchema(),
                           JsonNumberSchema(),
                           JsonNumberSchema()]
                ),
                spatial_res=JsonNumberSchema(
                    nullable=True,
                    exclusive_minimum=0.0),
                tile_size=JsonArraySchema(
                    nullable=True,
                    items=[
                        JsonIntegerSchema(minimum=1, maximum=2500),
                        JsonIntegerSchema(minimum=1, maximum=2500),
                    ]),
                time_range=JsonDateSchema.new_range(nullable=True),
                time_period=JsonStringSchema(
                    nullable=True,
                    pattern=r'^([1-9][0-9]*)?[DWMY]$'
                ),
            ),
            additional_properties=False,
            factory=cls
        )


def _to_dict(self, keys: Tuple[str, ...]) -> Dict[str, Any]:
    schema = self.get_schema()
    d = dict()
    for k in keys:
        v = getattr(self, k)
        if v is not None:
            s = schema.properties.get(k)
            if v != s.default:
                if isinstance(v, (str, bool, int, float)):
                    pass
                elif hasattr(v, 'to_dict') and callable(v.to_dict):
                    v = v.to_dict()
                elif isinstance(v, Mapping):
                    v = dict(v)
                elif isinstance(v, Iterable):
                    v = list(v)
                d[k] = v
    return d
