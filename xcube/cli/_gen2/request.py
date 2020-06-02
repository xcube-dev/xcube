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
from typing import Optional, Type, Dict, Any, Sequence, Mapping

import yaml

from xcube.cli._gen2.resample import RESAMPLE_PARAMS
from xcube.cli._gen2.transform import TRANSFORM_PARAMS
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema

input_config_schema = JsonObjectSchema(properties=dict(data_store_id=JsonStringSchema(),
                                                       dataset_id=JsonStringSchema()),
                                       required=['data_store_id', 'dataset_id'])
input_configs_schema = JsonArraySchema(items=input_config_schema)
cube_config_schema = RESAMPLE_PARAMS
code_config_schema = TRANSFORM_PARAMS
output_config_schema = JsonObjectSchema(properties=dict(data_store_id=JsonStringSchema(),
                                                        dataset_id=JsonStringSchema(default=None)),
                                        required=['data_store_id'])

request_schema = JsonObjectSchema(properties=dict(
    input_configs=input_configs_schema,
    cube_config=cube_config_schema,
    code_config=code_config_schema,
    output_config=output_config_schema)
)


# TODO: Refactor/rename properties. Currently modelled after xcube_sh requests.
# TODO: write tests

class InputConfig:
    def __init__(self, data_store_id: str, dataset_id: str, variable_names: Sequence[str]):
        self.data_store_id = data_store_id
        self.dataset_id = dataset_id
        self.variable_names = variable_names


input_config_schema.json_to_obj = InputConfig


class OutputConfig:
    def __init__(self, data_store_id: str, dataset_id: Optional[str]):
        self.data_store_id = data_store_id
        self.dataset_id = dataset_id


output_config_schema.json_to_obj = OutputConfig


class Request:
    # TODO: add fields
    def __init__(self,
                 input_configs: Sequence[InputConfig],
                 cube_config: Mapping[str, Any],
                 code_config: Mapping[str, Any],
                 output_config: OutputConfig):
        self.input_configs = input_configs
        self.cube_config = cube_config
        self.code_config = code_config
        self.output_config = output_config

    def to_dict(self) -> Mapping[str, Any]:
        """Convert into a JSON-serializable dictionary"""
        request_dict = request_schema.to_json_instance(self)
        request_schema.validate_instance(request_dict)
        return request_dict

    @classmethod
    def from_dict(cls, request_dict: Dict) -> 'Request':
        """Create new instance from a JSON-serializable dictionary"""
        request_schema.validate_instance(request_dict)
        return request_schema.from_json_instance(request_dict)

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


request_schema.json_to_obj = Request
# request_schema.obj_to_json = Request.from_dict
