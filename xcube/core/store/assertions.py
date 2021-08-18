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

from typing import Optional, Dict, Any, Callable

import jsonschema

from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonObjectSchema
from .error import DataStoreError


def assert_valid_params(params: Optional[Dict[str, Any]],
                        schema: Optional[JsonObjectSchema] = None,
                        name: str = 'params'):
    """
    Assert that constructor/method parameters *params* are valid.

    :param params: Dictionary of keyword arguments
        passed to a constructor/method.
    :param schema: The JSON Schema that *params* must adhere to.
    :param name: Name of the *params* variable.
    """
    _assert_valid(params, schema,
                  name, 'parameterization',
                  _validate_params)


def assert_valid_config(config: Optional[Dict[str, Any]],
                        schema: Optional[JsonObjectSchema] = None,
                        name: str = 'config'):
    """
    Assert that JSON object *config* is valid.

    :param config: JSON object used for configuration.
    :param schema: The JSON Schema that *config* must adhere to.
    :param name: Name of the *config* variable.
    """
    _assert_valid(config, schema,
                  name, 'configuration',
                  _validate_config)


def _validate_params(params: Dict[str, Any], schema: JsonObjectSchema):
    # Note, params is a dictionary of Python objects.
    # Convert them to a JSON instance
    # and perform JSON Schema validation
    schema.validate_instance(params)


def _validate_config(config: Dict[str, Any], schema: JsonObjectSchema):
    # Note, config is already a JSON object.
    # Perform JSON Schema validation directly.
    schema.validate_instance(config)


def _assert_valid(obj: Optional[Dict[str, Any]],
                  schema: Optional[JsonObjectSchema],
                  name: str,
                  kind: str,
                  validator: Callable[[Dict[str, Any],
                                       JsonObjectSchema], Any]):
    if obj is None:
        return
    assert_instance(obj, dict, name=name)
    if schema is not None:
        assert_instance(schema, JsonObjectSchema,
                        name=f'{name}_schema')
        try:
            validator(obj, schema)
        except jsonschema.ValidationError as e:
            raise DataStoreError(f'Invalid {kind}'
                                 f' detected: {e.message}') from e
