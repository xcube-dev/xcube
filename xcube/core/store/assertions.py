# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Optional, Dict, Any, Callable

import jsonschema

from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonObjectSchema
from .error import DataStoreError


def assert_valid_params(
    params: Optional[dict[str, Any]],
    schema: Optional[JsonObjectSchema] = None,
    name: str = "params",
):
    """Assert that constructor/method parameters *params* are valid.

    Args:
        params: Dictionary of keyword arguments passed to a
            constructor/method.
        schema: The JSON Schema that *params* must adhere to.
        name: Name of the *params* variable.
    """
    _assert_valid(params, schema, name, "parameterization", _validate_params)


def assert_valid_config(
    config: Optional[dict[str, Any]],
    schema: Optional[JsonObjectSchema] = None,
    name: str = "config",
):
    """Assert that JSON object *config* is valid.

    Args:
        config: JSON object used for configuration.
        schema: The JSON Schema that *config* must adhere to.
        name: Name of the *config* variable.
    """
    _assert_valid(config, schema, name, "configuration", _validate_config)


def _validate_params(params: dict[str, Any], schema: JsonObjectSchema):
    # Note, params is a dictionary of Python objects.
    # Convert them to a JSON instance
    # and perform JSON Schema validation
    schema.validate_instance(params)


def _validate_config(config: dict[str, Any], schema: JsonObjectSchema):
    # Note, config is already a JSON object.
    # Perform JSON Schema validation directly.
    schema.validate_instance(config)


def _assert_valid(
    obj: Optional[dict[str, Any]],
    schema: Optional[JsonObjectSchema],
    name: str,
    kind: str,
    validator: Callable[[dict[str, Any], JsonObjectSchema], Any],
):
    if obj is None:
        return
    assert_instance(obj, dict, name=name)
    if schema is not None:
        assert_instance(schema, JsonObjectSchema, name=f"{name}_schema")
        try:
            validator(obj, schema)
        except jsonschema.ValidationError as e:
            raise DataStoreError(f"Invalid {kind}" f" detected: {e.message}") from e
