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

import json
import os.path
import sys
from typing import Optional, Type, Dict, Any, Sequence, Mapping, Tuple

import yaml

from xcube.util.assertions import assert_condition
from xcube.util.assertions import assert_given
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema


class InputConfig:
    def __init__(self,
                 store_id: str = None,
                 opener_id: str = None,
                 data_id: str = None,
                 variable_names: Sequence[str] = None,
                 store_params: Mapping[str, Any] = None,
                 open_params: Mapping[str, Any] = None):
        assert_condition(store_id or opener_id, 'One of store_id and opener_id must be given')
        assert_given(data_id, 'data_id')
        self.store_id = store_id
        self.opener_id = opener_id
        self.data_id = data_id
        self.variable_names = variable_names
        self.store_params = store_params or {}
        self.open_params = open_params or {}

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                store_id=JsonStringSchema(min_length=1),
                opener_id=JsonStringSchema(min_length=1),
                data_id=JsonStringSchema(min_length=1),
                variable_names=JsonArraySchema(items=JsonStringSchema(min_length=1), min_items=1),
                store_params=JsonObjectSchema(),
                open_params=JsonObjectSchema()
            ),
            additional_properties=False,
            required=['data_id'],
            factory=cls,
        )

    def to_dict(self):
        d = dict()
        if self.store_id:
            d.update(store_id=str(self.store_id))
        if self.opener_id:
            d.update(writer_id=str(self.opener_id))
        if self.data_id:
            d.update(data_id=str(self.data_id))
        if self.variable_names:
            d.update(variable_names=list(self.variable_names))
        if self.store_params:
            d.update(store_params=dict(self.store_params))
        if self.open_params:
            d.update(open_params=dict(self.open_params))
        return d


class OutputConfig:

    def __init__(self,
                 store_id: str = None,
                 writer_id: str = None,
                 data_id: str = None,
                 store_params: Mapping[str, Any] = None,
                 write_params: Mapping[str, Any] = None):
        assert_condition(store_id or writer_id, 'One of store_id and writer_id must be given')
        self.store_id = store_id
        self.writer_id = writer_id
        self.data_id = data_id
        self.store_params = store_params or {}
        self.write_params = write_params or {}

    @classmethod
    def get_schema(cls):
        return JsonObjectSchema(
            properties=dict(
                store_id=JsonStringSchema(min_length=1),
                writer_id=JsonStringSchema(min_length=1),
                data_id=JsonStringSchema(default=None),
                store_params=JsonObjectSchema(),
                write_params=JsonObjectSchema(),
            ),
            additional_properties=False,
            required=[],
            factory=cls,
        )

    def to_dict(self):
        d = dict()
        if self.store_id:
            d.update(store_id=str(self.store_id))
        if self.writer_id:
            d.update(writer_id=str(self.writer_id))
        if self.data_id:
            d.update(data_id=str(self.data_id))
        if self.store_params:
            d.update(store_params=dict(self.store_params))
        if self.write_params:
            d.update(write_params=dict(self.write_params))
        return d


# Need to be aligned with params in resample_cube(cube, **params)
class CubeConfig:

    def __init__(self,
                 crs: str = None,
                 bbox: Tuple[float, float, float, float] = None,
                 spatial_res: float = None,
                 time_range: Tuple[str, Optional[str]] = None,
                 time_period: str = None):
        assert_given(bbox, 'bbox')
        assert_given(spatial_res, 'spatial_res')
        assert_given(time_range, 'time_range')
        self.crs = str(crs)
        self.bbox = tuple(bbox)
        self.spatial_res = float(spatial_res)
        self.time_range = tuple(time_range)
        self.time_period = str(time_period)

    def to_dict(self):
        d = dict(
            bbox=list(self.bbox),
            spatial_res=float(self.spatial_res),
            time_range=list(self.time_range)
        )
        if self.crs:
            d.update(crs=str(self.crs))
        if self.time_period:
            d.update(time_period=str(self.time_period))
        return d

    @classmethod
    def get_schema(cls):
        return JsonObjectSchema(
            properties=dict(
                crs=JsonStringSchema(nullable=True, default='WGS84', enum=[None, 'WGS84']),
                bbox=JsonArraySchema(items=[JsonNumberSchema(),
                                            JsonNumberSchema(),
                                            JsonNumberSchema(),
                                            JsonNumberSchema()]),
                spatial_res=JsonNumberSchema(exclusive_minimum=0.0),
                time_range=JsonArraySchema(items=[JsonStringSchema(format='date-time'),
                                                  JsonStringSchema(format='date-time', nullable=True)]),
                time_period=JsonStringSchema(nullable=True),
            ),
            additional_properties=True,
            required=['bbox', 'spatial_res', 'time_range'],
            factory=cls)


class Request:
    def __init__(self,
                 input_configs: Sequence[InputConfig] = None,
                 cube_config: Mapping[str, Any] = None,
                 output_config: OutputConfig = None):
        assert_given(input_configs, 'input_configs')
        assert_given(cube_config, 'cube_config')
        assert_given(output_config, 'output_config')
        self.input_configs = input_configs
        self.cube_config = cube_config
        self.output_config = output_config

    @classmethod
    def get_schema(cls):
        return JsonObjectSchema(
            properties=dict(
                input_configs=JsonArraySchema(items=InputConfig.get_schema(), min_items=1),
                cube_config=CubeConfig.get_schema(),
                output_config=OutputConfig.get_schema(),
            ),
            required=['input_configs', 'cube_config', 'output_config'],
            factory=cls,
        )

    def to_dict(self) -> Mapping[str, Any]:
        """Convert into a JSON-serializable dictionary"""
        return dict(input_configs=[ic.to_dict() for ic in self.input_configs],
                    cube_config=self.cube_config.to_dict(),
                    output_config=self.output_config.to_dict())

    @classmethod
    def from_dict(cls, request_dict: Dict) -> 'Request':
        """Create new instance from a JSON-serializable dictionary"""
        return cls.get_schema().from_instance(request_dict)

    @classmethod
    def from_file(cls, request_file: Optional[str], exception_type: Type[BaseException] = ValueError) -> 'Request':
        """Create new instance from a JSON file, or YAML file, or JSON passed via stdin."""
        request_dict = cls._load_request_file(request_file, exception_type=exception_type)
        return cls.from_dict(request_dict)

    @classmethod
    def _load_request_file(cls, request_file: Optional[str], exception_type: Type[BaseException] = ValueError) -> Dict:

        if request_file is not None and not os.path.exists(request_file):
            raise exception_type(f'Cube generation request "{request_file}" not found.')

        try:
            if request_file is None:
                if not sys.stdin.isatty():
                    return json.load(sys.stdin)
            else:
                with open(request_file, 'r') as fp:
                    if request_file.endswith('.json'):
                        return json.load(fp)
                    else:
                        return yaml.safe_load(fp)
        except BaseException as e:
            raise exception_type(f'Error loading cube generation request "{request_file}": {e}') from e

        raise exception_type(f'Missing cube generation request.')
