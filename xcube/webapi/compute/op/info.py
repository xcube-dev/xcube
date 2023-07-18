import inspect
from typing import Callable, Dict, Any
from mashumaro.jsonschema import build_json_schema

import xarray as xr

PyType = type(type(int))

# Functions registered in the operation registry will
# receive a new attribute with this name.
_ATTR_NAME_OP_INFO = '_op_info'

_PRIMITIVE_PY_TO_JSON_TYPES = {
    type(None): "null",
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
}

_PRIMITIVE_JSON_TO_PY_TYPES = {
    v: k for k, v in _PRIMITIVE_PY_TO_JSON_TYPES.items()
}


class OpInfo:
    def __init__(self,
                 params_schema: Dict[str, Any],
                 param_py_types: Dict[str, PyType]):
        self.params_schema = params_schema
        self.param_py_types = param_py_types

    def make_op(self, function: Callable) -> Callable:
        setattr(function, _ATTR_NAME_OP_INFO, self)
        return function

    @classmethod
    def get_op_info(cls, op: Callable) -> "OpInfo":
        op_info = getattr(op, _ATTR_NAME_OP_INFO, None)
        if not isinstance(op_info, OpInfo):
            raise ValueError(f"function {op.__name__}() is not yet"
                             f" registered as operation")
        return op_info

    @classmethod
    def new_op_info(cls, op: Callable) -> "OpInfo":
        members = dict(inspect.getmembers(op))
        annotations = members.get("__annotations__")
        code = members.get("__code__")
        param_py_types = {}
        params_schema = {}
        if code:
            args = inspect.getargs(code)
            required_param_names = set(args.args or []) \
                .union(set(args.varargs or []))
            optional_param_names = set(args.varkw or [])
            all_param_names = required_param_names.union(optional_param_names)
            if all_param_names:
                properties = {}
                for param_name in all_param_names:
                    py_type = annotations.get(param_name)
                    if py_type is not None:
                        param_py_types[param_name] = py_type
                    if py_type is xr.Dataset:
                        param_schema = {
                            "type": "string",
                            "title": "Dataset identifier"
                        }
                    elif py_type is not None:
                        # TODO: Decide what to do about mapping types with
                        # non-string keys (not supported in JSON)
                        param_schema = build_json_schema(py_type).to_dict()
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

        return OpInfo(params_schema, param_py_types)

    @property
    def effective_param_py_types(self) -> Dict[str, PyType]:
        py_types = self.param_py_types.copy()
        for param_name, param_schema in self.param_schemas.items():
            py_type = py_types.get(param_name)
            if py_type is None:
                json_type = param_schema.get("type")
                py_type = _PRIMITIVE_JSON_TO_PY_TYPES.get(json_type)
                if py_type is not None:
                    py_types[param_name] = py_type
        return py_types

    @property
    def param_schemas(self) -> Dict[str, Any]:
        return self.params_schema.get("properties", {})

    def set_param_schemas(self, schemas: Dict[str, Any]):
        self.params_schema["properties"] = schemas

    def update_params_schema(self, schema: Dict[str, Any]):
        self.params_schema.update(schema)

    def update_param_schema(self, param_name: str, schema: Dict[str, Any]):
        param_schemas = self.param_schemas
        param_schema = param_schemas.get(param_name, {})
        param_schema.update(schema)
        param_schemas[param_name] = param_schema
        self.set_param_schemas(param_schemas)

    def get_param_py_type(self, param_name: str):
        return self.param_py_types[param_name]

    def set_param_py_type(self, param_name: str, py_type: PyType):
        self.param_py_types[param_name] = py_type
