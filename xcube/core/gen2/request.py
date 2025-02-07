# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import json
import os.path
import sys
from collections.abc import Sequence
from typing import Any, Dict, Optional, Union

import jsonschema
import yaml

from xcube.core.byoa import CodeConfig
from xcube.util.assertions import assert_false, assert_instance, assert_true
from xcube.util.jsonschema import JsonArraySchema, JsonObject, JsonObjectSchema

from ...constants import LOG
from .config import CallbackConfig, CubeConfig, InputConfig, OutputConfig
from .error import CubeGeneratorError

CubeGeneratorRequestLike = Union[str, dict, "CubeGeneratorRequest"]


class CubeGeneratorRequest(JsonObject):
    """
    A request used to generate data cubes using cube generators.

    Args:
        input_config: A configuration for a single input.
            Must be omitted if *input_configs* is given.
        input_configs: A sequence of one or more input configurations.
            Must be omitted if *input_config* is given.
        cube_config: The target cube configuration.
        code_config: The user-code configuration.
        output_config: The output configuration for the target cube.
        callback_config: A configuration that allows a cube generator
            to publish progress information to a compatible endpoint.
    """

    def __init__(
        self,
        input_config: InputConfig = None,
        input_configs: Sequence[InputConfig] = None,
        cube_config: CubeConfig = None,
        code_config: CodeConfig = None,
        output_config: OutputConfig = None,
        callback_config: Optional[CallbackConfig] = None,
    ):
        assert_true(
            input_config or input_configs,
            "one of input_config and input_configs must be given",
        )
        assert_false(
            input_config and input_configs,
            "input_config and input_configs cannot be given both",
        )
        if input_config is not None:
            assert_instance(input_config, InputConfig, "input_config")
            input_configs = [input_config]
        elif input_configs is not None:
            assert_instance(input_configs, (list, tuple), "input_configs")
            for i in range(len(input_configs)):
                assert_instance(input_configs[i], InputConfig, f"input_configs[{i}]")
        if cube_config is not None:
            assert_instance(cube_config, CubeConfig, "cube_config")
        if code_config is not None:
            assert_instance(code_config, CodeConfig, "code_config")
        assert_instance(output_config, OutputConfig, "output_config")
        if callback_config is not None:
            assert_instance(callback_config, CallbackConfig, "callback_config")
        self.input_configs = input_configs
        self.cube_config = cube_config
        self.code_config = code_config
        self.output_config = output_config
        self.callback_config = callback_config

    def for_service(self) -> "CubeGeneratorRequest":
        if self.code_config is None:
            return self
        return CubeGeneratorRequest(
            input_configs=self.input_configs,
            cube_config=self.cube_config,
            code_config=self.code_config.for_service(),
            output_config=self.output_config,
            callback_config=self.callback_config,
        )

    def for_local(self) -> "CubeGeneratorRequest":
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
                input_configs=JsonArraySchema(
                    items=InputConfig.get_schema(), min_items=1
                ),
                cube_config=CubeConfig.get_schema(),
                code_config=CodeConfig.get_schema(),
                output_config=OutputConfig.get_schema(),
                callback_config=CallbackConfig.get_schema(),
            ),
            required=["output_config"],
            factory=cls,
        )

    @classmethod
    def normalize(cls, request: CubeGeneratorRequestLike) -> "CubeGeneratorRequest":
        """Normalize given *request* to an instance of
        :class:`CubeGeneratorRequest`.

        If *request* is already a CubeGeneratorRequest it is returned as is.
        If it is a ``str``, it is interpreted as a YAML or JSON file path
        and the request is read from file using
        ``CubeGeneratorRequest.from_file()``.
        If it is a ``dict``, it is interpreted as a JSON object and the
        request is parsed using ``CubeGeneratorRequest.from_dict()``.

        Args:
            request: The request, or request file path,
                or request JSON object.
        Raises:
            TypeError: if *request* is not a ``CubeGeneratorRequest``,
                ``str``, or ``dict``.
        """
        if isinstance(request, CubeGeneratorRequest):
            return request
        if isinstance(request, str):
            return CubeGeneratorRequest.from_file(request)
        if isinstance(request, dict):
            return CubeGeneratorRequest.from_dict(request)
        raise TypeError(
            "request must be a str, dict, or a CubeGeneratorRequest instance"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert into a JSON-serializable dictionary"""
        d = {}

        if self.input_configs is not None:
            if len(self.input_configs) == 1:
                d.update(input_config=self.input_configs[0].to_dict())
            else:
                d.update(input_configs=[ic.to_dict() for ic in self.input_configs])

        if self.cube_config is not None:
            d.update(cube_config=self.cube_config.to_dict())

        if self.code_config is not None:
            d.update(code_config=self.code_config.to_dict())

        d.update(output_config=self.output_config.to_dict())

        if self.callback_config is not None:
            d.update(callback_config=self.callback_config.to_dict())

        return d

    @classmethod
    def from_dict(cls, request_dict: dict[str, Any]) -> "CubeGeneratorRequest":
        """Create new instance from a JSON-serializable dictionary"""
        try:
            return cls.get_schema().from_instance(request_dict)
        except jsonschema.exceptions.ValidationError as e:
            raise CubeGeneratorError(f"{e}", status_code=400) from e

    @classmethod
    def from_file(
        cls, request_file: Optional[str], verbosity: int = 0
    ) -> "CubeGeneratorRequest":
        """
        Create new instance from a JSON file, or YAML file,
        or JSON passed via stdin.
        """
        request_dict = cls._load_request_file(request_file, verbosity=verbosity)
        if verbosity:
            LOG.info(f"Cube generator request loaded from {request_file or 'TTY'}.")
        return cls.from_dict(request_dict)

    @classmethod
    def _load_request_file(
        cls, gen_config_file: Optional[str], verbosity: int = 0
    ) -> dict:
        if gen_config_file is not None and not os.path.exists(gen_config_file):
            raise CubeGeneratorError(
                f'Cube generator request "{gen_config_file}" not found.'
            )

        try:
            if gen_config_file is None:
                if not sys.stdin.isatty():
                    if verbosity:
                        LOG.info("Awaiting generator request JSON from TTY...")
                    return json.load(sys.stdin)
            else:
                with open(gen_config_file) as fp:
                    if gen_config_file.endswith(".json"):
                        return json.load(fp)
                    else:
                        return yaml.safe_load(fp)
        except BaseException as e:
            raise CubeGeneratorError(
                f'Error loading generator request "{gen_config_file}": {e}'
            ) from e

        raise CubeGeneratorError(f"Missing cube generator request.")
