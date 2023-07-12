import inspect
import warnings
from typing import Callable, Dict, Optional, Union, Any

import xarray as xr

from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonSchema

from xcube.util.jsonschema import JsonObjectSchema

_OP_REGISTRY: Dict[str, Callable] = {}
_OP_PARAMS_SCHEMA_ATTR_NAME = '_params_schema'

_PRIMITIVE_PY_TO_JSON_TYPES = {
    type(None): "null",
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
}


def register_op(f: Callable):
    f_name = f.__name__
    prev_f = _OP_REGISTRY.get(f_name)
    if prev_f is None:
        _OP_REGISTRY[f.__name__] = f
        set_op_params_schema(f, compute_params_schema(f))
    elif prev_f is not f:
        warnings.warn(f'redefining already registered operation {f_name!r}')


def compute_params_schema(f: Callable):
    members = dict(inspect.getmembers(f))
    annotations = members.get("__annotations__")
    code = members.get("__code__")
    params_schema = {}
    if code:
        args = inspect.getargs(code)
        required_param_names = set(args.args or [])\
            .union(set(args.varargs or []))
        optional_param_names = set(args.varkw or [])
        all_param_names = required_param_names.union(optional_param_names)
        if all_param_names:
            properties = {}
            for param_name in all_param_names:
                py_type = annotations.get(param_name)
                # print(param_name, "-------------->",
                #       py_type, type(py_type), flush=True)
                if py_type is xr.Dataset:
                    param_schema = {
                        "type": "string",
                        "title": "Dataset identifier"
                    }
                elif py_type is not None:
                    json_type = _PRIMITIVE_PY_TO_JSON_TYPES.get(py_type)
                    if json_type is None:
                        # TODO: decode json_type
                        json_type = repr(py_type)
                    param_schema = {
                        "type": json_type,
                    }
                else:
                    param_schema = {}
                properties[param_name] = param_schema
            params_schema = {
                "type": "object",
                "properties": properties,
                "required": list(required_param_names),
                "additionalProperties": False
            }
        else:
            params_schema.update({
                "type": ["null", "object"],
                "additionalProperties": False
            })

    return params_schema


def get_op_params_schema(f: Callable) -> Dict[str, any]:
    return getattr(f, _OP_PARAMS_SCHEMA_ATTR_NAME, {}).copy()


def set_op_params_schema(
        f: Callable,
        params_schema: Union[JsonObjectSchema, Dict[str, any]]
):
    assert_instance(params_schema, (JsonObjectSchema, dict),
                    name='params_schema')
    if isinstance(params_schema, JsonObjectSchema):
        params_schema = params_schema.to_dict()
    setattr(f, _OP_PARAMS_SCHEMA_ATTR_NAME, params_schema)


def get_operations() -> Dict[str, Callable]:
    return _OP_REGISTRY.copy()


def assert_decorator_target_ok(decorator_name: str, target: Any):
    if not callable(target):
        raise TypeError(f"decorator {decorator_name!r}"
                        f" can be used with callables only")


def op(_func: Optional[Callable] = None,
       title: Optional[str] = None,
       description: Optional[str] = None,
       params_schema: Optional[JsonObjectSchema] = None):
    def decorator(f: Callable):
        assert_decorator_target_ok("op", f)
        register_op(f)
        if params_schema is not None:
            set_op_params_schema(f, params_schema)
        return f

    if _func is None:
        return decorator
    else:
        return decorator(_func)


# See also
# https://json-schema.org/draft/2020-12/json-schema-validation.html#name-a-vocabulary-for-basic-meta

def op_param(name: str,
             title: Optional[str] = None,
             description: Optional[str] = None,
             default: Optional[Any] = None,
             required: Optional[bool] = None,
             schema: Optional[JsonSchema] = None):
    """Decorator that adds schema information for the parameter
    given by *name*.

    See also
    https://json-schema.org/draft/2020-12/json-schema-validation.html#name-a-vocabulary-for-basic-meta
    """
    def decorator(f: Callable):
        assert_decorator_target_ok("op", f)
        register_op(f)
        if schema is not None:
            _update_param_schema(f, name, schema.to_dict())
        if title is not None:
            _update_param_schema(f, name, {"title": title})
        if description is not None:
            _update_param_schema(f, name, {"description": description})
        if default is not None:
            _update_param_schema(f, name, {"default": default})
        if required is not None:
            params_schema = get_op_params_schema(f)
            required_set = set(params_schema.get("required", []))
            if required and name not in required_set:
                required_set.add(name)
            elif not required and name in required_set:
                required_set.remove(name)
            params_schema["required"] = list(required_set)
            set_op_params_schema(f, params_schema)

        return f

    return decorator


def _update_param_schema(f: Callable, name: str, value: Dict[str, Any]):
    params_schema = get_op_params_schema(f)
    properties = params_schema.get("properties", {}).copy()
    param_schema = properties.get(name, {}).copy()
    param_schema.update(value)
    properties[name] = param_schema
    params_schema["properties"] = properties
    set_op_params_schema(f, params_schema)


