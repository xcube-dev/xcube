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

from typing import Any, Optional, Callable, Type, Dict

import jsonschema
import xarray as xr

from xcube.core.byoa import CodeConfig
from xcube.core.gridmapping import GridMapping
from xcube.util.jsonschema import JsonObjectSchema
from .transformer import CubeTransformer
from .transformer import TransformedCube
from ..config import CubeConfig
from ..error import CubeGeneratorError
from ..processor import DatasetProcessor
from ..processor import METHOD_NAME_DATASET_PROCESSOR
from ..processor import METHOD_NAME_PARAMS_SCHEMA_GETTER


class CubeUserCodeExecutor(CubeTransformer):
    """Execute user code."""

    def __init__(self, code_config: CodeConfig):
        user_code_callable = code_config.get_callable()
        user_code_callable_params = code_config.callable_params or {}

        if isinstance(user_code_callable, type):
            user_code_callable = self._get_callable_from_class(
                user_code_callable,
                user_code_callable_params
            )

        self._callable = user_code_callable
        self._callable_params = user_code_callable_params

    def transform_cube(self,
                       cube: xr.Dataset,
                       gm: GridMapping,
                       cube_config: CubeConfig) -> TransformedCube:
        return self._callable(cube, **self._callable_params), gm, cube_config

    @classmethod
    def _get_callable_from_class(
            cls,
            process_class: Type[Any],
            process_params: Dict[str, Any],
    ) -> Callable:
        process_object = process_class()
        process_params_schema = None
        if isinstance(process_object, DatasetProcessor):
            process_callable = process_object.process_dataset
            process_params_schema = process_object.get_process_params_schema()
        else:
            process_callable = cls._get_user_code_callable(
                process_object,
                METHOD_NAME_DATASET_PROCESSOR,
                require=True
            )
            process_params_schema_getter = cls._get_user_code_callable(
                process_object,
                METHOD_NAME_PARAMS_SCHEMA_GETTER,
                require=False
            )
            if process_params_schema_getter is not None:
                process_params_schema = process_params_schema_getter()
        if process_params_schema is not None:
            if not isinstance(process_params_schema, JsonObjectSchema):
                raise CubeGeneratorError(
                    f'Parameter schema returned by'
                    f' user code class {process_class!r}'
                    f' must be an instance of {JsonObjectSchema!r}',
                    status_code=400
                )
            try:
                process_params_schema.validate_instance(process_params)
            except jsonschema.ValidationError as e:
                raise CubeGeneratorError(
                    f'Invalid processing parameters: {e}',
                    status_code=400
                ) from e
        return process_callable

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
                f' in user code class {type(user_code_object)!r}',
                status_code=400
            )
        user_code_callable = getattr(user_code_object, method_name)
        if not callable(user_code_callable):
            raise CubeGeneratorError(
                f'Attribute {METHOD_NAME_DATASET_PROCESSOR!r}'
                f' of user code class {type(user_code_object)!r}'
                f' must be callable',
                status_code=400
            )
        return user_code_callable
