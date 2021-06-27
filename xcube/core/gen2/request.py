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

import json
import os.path
import sys
from typing import Optional, Dict, Any, Sequence

import jsonschema
import yaml

from xcube.core.byoa import CodeConfig
from xcube.util.assertions import assert_condition
from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonObject
from xcube.util.jsonschema import JsonObjectSchema
from .config import CallbackConfig
from .config import CubeConfig
from .config import InputConfig
from .config import OutputConfig
from .error import CubeGeneratorError


class CubeGeneratorRequest(JsonObject):
    """
    A request used to generate data cubes using cube generators.

    :param input_config: A configuration for a single input.
        Must be omitted if *input_configs* is given.
    :param input_configs: A sequence of one or more input configurations.
        Must be omitted if *input_config* is given.
    :param cube_config: The target cube configuration.
    :param code_config: The user-code configuration.
    :param output_config: The output configuration for the target cube.
    :param callback_config: A configuration that allows a cube generator
        to publish progress information to a compatible endpoint.
    """

    def __init__(self,
                 input_config: InputConfig = None,
                 input_configs: Sequence[InputConfig] = None,
                 cube_config: CubeConfig = None,
                 code_config: CodeConfig = None,
                 output_config: OutputConfig = None,
                 callback_config: Optional[CallbackConfig] = None):
        assert_condition(input_config or input_configs,
                         'one of input_config and input_configs must be given')
        assert_condition(not (input_config and input_configs),
                         'input_config and input_configs cannot be given both')
        if input_config is not None:
            assert_instance(input_config, InputConfig, 'input_config')
            input_configs = [input_config]
        elif input_configs is not None:
            assert_instance(input_configs, (list, tuple), 'input_configs')
            for i in range(len(input_configs)):
                assert_instance(input_configs[i], InputConfig,
                                f'input_configs[{i}]')
        assert_instance(cube_config, CubeConfig, 'cube_config')
        if code_config is not None:
            assert_instance(code_config, CodeConfig, 'code_config')
        assert_instance(output_config, OutputConfig, 'output_config')
        if callback_config is not None:
            assert_instance(callback_config, CallbackConfig, 'callback_config')
        self.input_configs = input_configs
        self.cube_config = cube_config
        self.code_config = code_config
        self.output_config = output_config
        self.callback_config = callback_config

    def for_service(self) -> 'CubeGeneratorRequest':
        if self.code_config is None:
            return self
        return CubeGeneratorRequest(
            input_configs=self.input_configs,
            cube_config=self.cube_config,
            code_config=self.code_config.for_service(),
            output_config=self.output_config,
            callback_config=self.callback_config,
        )

    def for_local(self) -> 'CubeGeneratorRequest':
        if self.code_config is None:
            return self
        return CubeGeneratorRequest(
            input_configs=self.input_configs,
            cube_config=self.cube_config,
            code_config=self.code_config.for_local(),
            output_config=self.output_config,
            callback_config=self.callback_config,
        )

    @classmethod
    def get_schema(cls):
        return JsonObjectSchema(
            properties=dict(
                input_config=InputConfig.get_schema(),
                input_configs=JsonArraySchema(items=InputConfig.get_schema(), min_items=1),
                cube_config=CubeConfig.get_schema(),
                code_config=CodeConfig.get_schema(),
                output_config=OutputConfig.get_schema(),
                callback_config=CallbackConfig.get_schema()
            ),
            required=['cube_config', 'output_config'],
            factory=cls,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert into a JSON-serializable dictionary"""
        d = {}

        if self.input_configs is not None:
            if len(self.input_configs) == 1:
                d.update(input_config=self.input_configs[0].to_dict())
            else:
                d.update(input_configs=[ic.to_dict() for ic in self.input_configs])

        d.update(cube_config=self.cube_config.to_dict(),
                 output_config=self.output_config.to_dict())

        if self.code_config is not None:
            d.update(code_config=self.code_config.to_dict())

        if self.callback_config is not None:
            d.update(callback_config=self.callback_config.to_dict())

        return d

    @classmethod
    def from_dict(cls, request_dict: Dict[str, Any]) -> 'CubeGeneratorRequest':
        """Create new instance from a JSON-serializable dictionary"""
        try:
            return cls.get_schema().from_instance(request_dict)
        except jsonschema.exceptions.ValidationError as e:
            raise CubeGeneratorError(f'{e}') from e

    @classmethod
    def from_file(cls, request_file: Optional[str], verbosity: int = 0) -> 'CubeGeneratorRequest':
        """Create new instance from a JSON file, or YAML file, or JSON passed via stdin."""
        gen_config_dict = cls._load_gen_config_file(request_file, verbosity=verbosity)
        if verbosity:
            print(f'Cube generator configuration loaded from {request_file or "TTY"}.')
        return cls.from_dict(gen_config_dict)

    @classmethod
    def _load_gen_config_file(cls, gen_config_file: Optional[str], verbosity: int = 0) -> Dict:

        if gen_config_file is not None and not os.path.exists(gen_config_file):
            raise CubeGeneratorError(f'Cube generator configuration "{gen_config_file}" not found.')

        try:
            if gen_config_file is None:
                if not sys.stdin.isatty():
                    if verbosity:
                        print('Awaiting generator configuration JSON from TTY...')
                    return json.load(sys.stdin)
            else:
                with open(gen_config_file, 'r') as fp:
                    if gen_config_file.endswith('.json'):
                        return json.load(fp)
                    else:
                        return yaml.safe_load(fp)
        except BaseException as e:
            raise CubeGeneratorError(f'Error loading generator configuration "{gen_config_file}": {e}') from e

        raise CubeGeneratorError(f'Missing cube generator configuration.')
