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
from typing import Any, Optional, Callable

import xarray as xr

from xcube.core.byoa import CodeConfig
from .error import CubeGeneratorError
from .processor import CubeProcessor
from ...util.jsonschema import JsonObjectSchema

_CLASS_METHOD_NAME_PROCESS_DATASET = 'process_dataset'
_CLASS_METHOD_NAME_GET_PARAMS_SCHEMA = 'get_params_schema'


class CubeCodeExecutor(CubeProcessor):
    """Execute user code."""

    def __init__(self, code_config: CodeConfig):
        user_code_callable = code_config.get_callable()
        user_code_callable_params = code_config.callable_params or {}

        if isinstance(user_code_callable, type):
            user_code_callable = self._get_user_code_callable_from_class(
                user_code_callable,
                user_code_callable_params
            )

        self._callable = user_code_callable
        self._callable_params = user_code_callable_params

    def process_cube(self, cube: xr.Dataset) -> xr.Dataset:
        return self._callable(cube, **self._callable_params)

    @classmethod
    def _get_user_code_callable_from_class(cls, user_code_callable, user_code_callable_params):
        user_code_class = user_code_callable
        user_code_object = user_code_class()
        user_code_callable = cls._get_user_code_callable(
            user_code_object,
            _CLASS_METHOD_NAME_PROCESS_DATASET,
            require=True
        )
        params_schema_getter = cls._get_user_code_callable(
            user_code_object,
            _CLASS_METHOD_NAME_GET_PARAMS_SCHEMA,
            require=False
        )
        if params_schema_getter:
            params_schema = params_schema_getter()
            if not isinstance(params_schema, JsonObjectSchema):
                raise CubeGeneratorError(
                    f'Parameter schema returned by method'
                    f' {_CLASS_METHOD_NAME_GET_PARAMS_SCHEMA!r}'
                    f' of user code class {user_code_class!r}'
                    f' must be an instance of JsonObjectSchema')
            params_schema.validate_instance(user_code_callable_params)
        return user_code_callable

    @classmethod
    def _get_user_code_callable(cls,
                                user_code_object: Any,
                                method_name: str,
                                require: bool = True) -> Optional[Callable]:
        if not hasattr(user_code_object,
                       method_name):
            if not require:
                return None
            raise CubeGeneratorError(
                f'Missing method {method_name!r}'
                f' in user code class {type(user_code_object)!r}'
            )
        user_code_callable = getattr(user_code_object, method_name)
        if not callable(user_code_callable):
            raise CubeGeneratorError(
                f'Attribute {_CLASS_METHOD_NAME_PROCESS_DATASET!r}'
                f' of user code class {type(user_code_object)!r}'
                f' must be callable'
            )
        return user_code_callable
