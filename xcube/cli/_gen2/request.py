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
import yaml
from typing import Optional, Type, Dict, Any, Sequence, Mapping, Tuple

from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema


# TODO: Refactor/rename properties. Currently modelled after xcube_sh requests.

class InputConfig:
    def __init__(self,
                 cube_store_id: str,
                 cube_id: str,
                 variable_names: Sequence[str],
                 cube_store_params: Mapping[str, Any] = None,
                 open_params: Mapping[str, Any] = None):
        self.cube_store_id = cube_store_id
        self.cube_id = cube_id
        self.variable_names = variable_names
        self.cube_store_params = cube_store_params
        self.open_params = open_params

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                cube_store_id=JsonStringSchema(min_length=1),
                cube_id=JsonStringSchema(min_length=1),
                variable_names=JsonArraySchema(items=JsonStringSchema(min_length=1), min_items=1),
                cube_store_params=JsonObjectSchema(),
                open_params=JsonObjectSchema()
            ),
            additional_properties=False,
            required=['cube_store_id', 'cube_id', 'variable_names'],
            factory=cls,
        )


class OutputConfig:

    def __init__(self,
                 cube_store_id: str,
                 cube_id: str = None,
                 cube_store_params: Mapping[str, Any] = None,
                 write_params: Mapping[str, Any] = None):
        self.cube_store_id = cube_store_id
        self.cube_id = cube_id
        self.cube_store_params = cube_store_params
        self.write_params = write_params

    @classmethod
    def get_schema(cls):
        return JsonObjectSchema(
            properties=dict(
                cube_store_id=JsonStringSchema(min_length=1),
                cube_id=JsonStringSchema(default=None),
                cube_store_params=JsonObjectSchema(),
                write_params=JsonObjectSchema(),
            ),
            additional_properties=False,
            required=['cube_store_id'],
            factory=cls,
        )


# Need to be aligned with params in resample_cube(cube, **params)
class CubeConfig:

    def __init__(self,
                 spatial_crs: Optional[str],
                 spatial_coverage: Tuple[float, float, float, float],
                 spatial_resolution: float,
                 temporal_coverage: Tuple[str, Optional[str]],
                 temporal_resolution: str = None):
        self.spatial_crs = spatial_crs
        self.spatial_coverage = spatial_coverage
        self.spatial_resolution = spatial_resolution
        self.temporal_coverage = temporal_coverage
        self.temporal_resolution = temporal_resolution

    @classmethod
    def get_schema(cls):
        return JsonObjectSchema(
            properties=dict(
                spatial_crs=JsonStringSchema(nullable=True, default='WGS84', enum=[None, 'WGS84']),
                spatial_coverage=JsonArraySchema(items=[JsonNumberSchema(),
                                                        JsonNumberSchema(),
                                                        JsonNumberSchema(),
                                                        JsonNumberSchema()]),
                spatial_resolution=JsonNumberSchema(exclusive_minimum=0.0),
                temporal_coverage=JsonArraySchema(items=[JsonStringSchema(format='date-time'),
                                                         JsonStringSchema(format='date-time', nullable=True)]),
                temporal_resolution=JsonStringSchema(nullable=True),
            ),
            additional_properties=True,
            required=['spatial_coverage', 'spatial_resolution', 'temporal_coverage'],
            factory=cls)


class GitHubConfig:
    def __init__(self,
                 repo_name: str,
                 user_name: str,
                 access_token: str):
        self.repo_name = repo_name
        self.user_name = user_name
        self.access_token = access_token

    @classmethod
    def get_schema(cls):
        return JsonObjectSchema(
            properties=dict(
                repo_name=JsonStringSchema(min_length=1),
                user_name=JsonStringSchema(min_length=1),
                access_token=JsonStringSchema(min_length=1),
            ),
            additional_properties=False,
            required=['repo_name', 'user_name', 'access_token'],
            factory=cls,
        )


# Need to be aligned with params in transform_cube(cube, **params)
class CodeConfig:
    def __init__(self,
                 python_code: Optional[str] = None,
                 git_hub: Optional[str] = None,
                 function_name: str = None,
                 function_params: Mapping[str, Any] = None):
        self.python_code = python_code
        self.git_hub = git_hub
        self.function_name = function_name
        self.function_params = function_params

    @classmethod
    def get_schema(cls):
        return JsonObjectSchema(
            properties=dict(
                python_code=JsonStringSchema(),
                git_hub=GitHubConfig.get_schema(),
                function_name=JsonStringSchema(),
                function_params=JsonObjectSchema(),
            ),
            additional_properties=False,
            factory=cls,
        )


class Request:
    def __init__(self,
                 input_configs: Sequence[InputConfig],
                 cube_config: Mapping[str, Any],
                 output_config: OutputConfig,
                 code_config: Optional[Mapping[str, Any]] = None):
        self.input_configs = input_configs
        self.cube_config = cube_config
        self.code_config = code_config
        self.output_config = output_config

    @classmethod
    def get_schema(cls):
        return JsonObjectSchema(
            properties=dict(
                input_configs=JsonArraySchema(items=InputConfig.get_schema(), min_items=1),
                cube_config=CubeConfig.get_schema(),
                code_config=CodeConfig.get_schema(),
                output_config=OutputConfig.get_schema(),
            ),
            required=['input_configs', 'cube_config', 'output_config'],
            factory=cls,
        )

    def to_dict(self) -> Mapping[str, Any]:
        """Convert into a JSON-serializable dictionary"""
        return self.get_schema().to_instance(self)

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
