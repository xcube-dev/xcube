# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Any, Callable, Dict, Optional, Type

import jsonschema
import xarray as xr

from xcube.core.byoa import CodeConfig
from xcube.core.gridmapping import GridMapping
from xcube.util.jsonschema import JsonObjectSchema

from ..config import CubeConfig
from ..error import CubeGeneratorError
from ..processor import (
    METHOD_NAME_DATASET_PROCESSOR,
    METHOD_NAME_PARAMS_SCHEMA_GETTER,
    DatasetProcessor,
)
from .transformer import CubeTransformer, TransformedCube


class CubeUserCodeExecutor(CubeTransformer):
    """Execute user code."""

    def __init__(self, code_config: CodeConfig):
        user_code_callable = code_config.get_callable()
        user_code_callable_params = code_config.callable_params or {}

        if isinstance(user_code_callable, type):
            user_code_callable = self._get_callable_from_class(
                user_code_callable, user_code_callable_params
            )

        self._callable = user_code_callable
        self._callable_params = user_code_callable_params

    def transform_cube(
        self, cube: xr.Dataset, gm: GridMapping, cube_config: CubeConfig
    ) -> TransformedCube:
        return self._callable(cube, **self._callable_params), gm, cube_config

    @classmethod
    def _get_callable_from_class(
        cls,
        process_class: type[Any],
        process_params: dict[str, Any],
    ) -> Callable:
        process_object = process_class()
        process_params_schema = None
        if isinstance(process_object, DatasetProcessor):
            process_callable = process_object.process_dataset
            process_params_schema = process_object.get_process_params_schema()
        else:
            process_callable = cls._get_user_code_callable(
                process_object, METHOD_NAME_DATASET_PROCESSOR, require=True
            )
            process_params_schema_getter = cls._get_user_code_callable(
                process_object, METHOD_NAME_PARAMS_SCHEMA_GETTER, require=False
            )
            if process_params_schema_getter is not None:
                process_params_schema = process_params_schema_getter()
        if process_params_schema is not None:
            if not isinstance(process_params_schema, JsonObjectSchema):
                raise CubeGeneratorError(
                    f"Parameter schema returned by"
                    f" user code class {process_class!r}"
                    f" must be an instance of {JsonObjectSchema!r}",
                    status_code=400,
                )
            try:
                process_params_schema.validate_instance(process_params)
            except jsonschema.ValidationError as e:
                raise CubeGeneratorError(
                    f"Invalid processing parameters: {e}", status_code=400
                ) from e
        return process_callable

    @classmethod
    def _get_user_code_callable(
        cls, user_code_object: Any, method_name: str, require: bool = True
    ) -> Optional[Callable]:
        if not hasattr(user_code_object, method_name):
            if not require:
                return None
            raise CubeGeneratorError(
                f"Missing method {method_name!r}"
                f" in user code class {type(user_code_object)!r}",
                status_code=400,
            )
        user_code_callable = getattr(user_code_object, method_name)
        if not callable(user_code_callable):
            raise CubeGeneratorError(
                f"Attribute {METHOD_NAME_DATASET_PROCESSOR!r}"
                f" of user code class {type(user_code_object)!r}"
                f" must be callable",
                status_code=400,
            )
        return user_code_callable
