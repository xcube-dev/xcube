# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Any, Callable, List, Optional, Union

from xcube.util.jsonschema import JsonObjectSchema, JsonSchema

from .info import PyType
from .registry import OP_REGISTRY, OpRegistry


def operation(
    _op: Optional[Callable] = None,
    params_schema: Optional[JsonObjectSchema] = None,
    op_registry: OpRegistry = OP_REGISTRY,
):
    """Decorator that registers a function as an operation.

    Args:
        _op: the function to register as an operation
        params_schema: JSON Schema for the function's parameters (a JSON
            object schema mapping parameter names to their individual
            schemas)
        op_registry: the registry in which to register the operation

    Returns:
        the decorated operation, if an operation was supplied;
        otherwise, a decorator function
    """

    def decorator(op: Callable):
        _assert_decorator_target_ok("operation", op)
        op_info = op_registry.register_op(op)
        if params_schema is not None:
            op_info.update_params_schema(params_schema.to_dict())
        return op

    if _op is None:
        return decorator
    else:
        return decorator(_op)


def op_param(
    name: str,
    json_type: Optional[Union[str, list[str]]] = None,
    py_type: Optional[PyType] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    default: Optional[Any] = None,
    required: Optional[bool] = None,
    schema: Optional[JsonSchema] = None,
    op_registry: OpRegistry = OP_REGISTRY,
):
    """Decorator that adds schema information to the operation parameter given
    by *name*.

    See also
    https://json-schema.org/draft/2020-12/json-schema-validation.html#name-a-vocabulary-for-basic-meta

    Args:
        name: name of the parameter to apply schema information to
        json_type: JSON Schema type of the parameter
        py_type: Python type of the parameter
        title: title of the parameter
        description: description of the parameter
        default: default value for the parameter
        required: whether the parameter is required
        schema: JSON Schema describing the parameter
        op_registry: registry in which to register the operation

    Returns:
        parameterized decorator for a compute operation function
    """

    def decorator(op: Callable):
        _assert_decorator_target_ok("op_param", op)
        op_info = op_registry.register_op(op)
        op_param_schema = {}
        if schema is not None:
            op_param_schema.update(schema.to_dict())
        if py_type is not None:
            op_info.set_param_py_type(name, py_type)
        if json_type is not None:
            op_param_schema.update({"type": json_type})
        if title is not None:
            op_param_schema.update({"title": title})
        if description is not None:
            op_param_schema.update({"description": description})
        if default is not None:
            op_param_schema.update({"default": default})
        if op_param_schema:
            op_info.update_param_schema(name, op_param_schema)
        if required is not None:
            required_set = set(op_info.params_schema.get("required", []))
            if required and name not in required_set:
                required_set.add(name)
            elif not required and name in required_set:
                required_set.remove(name)
            op_info.update_params_schema({"required": list(required_set)})
        return op

    return decorator


def _assert_decorator_target_ok(decorator_name: str, target: Any):
    if not callable(target):
        raise TypeError(f"decorator {decorator_name!r} can be used with callables only")
