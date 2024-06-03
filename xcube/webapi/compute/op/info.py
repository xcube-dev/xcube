# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import collections.abc
import inspect
import typing
from typing import Callable, Dict, Any
from mashumaro.jsonschema import build_json_schema

import xarray as xr

from xcube.core.mldataset import MultiLevelDataset

PyType = type(type(int))

# Functions registered in the operation registry will
# receive a new attribute with this name.
_ATTR_NAME_OP_INFO = "_op_info"

_PRIMITIVE_PY_TO_JSON_TYPES = {
    type(None): "null",
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
}

_PRIMITIVE_JSON_TO_PY_TYPES = {v: k for k, v in _PRIMITIVE_PY_TO_JSON_TYPES.items()}


class OpInfo:
    """Information about a compute operation"""

    def __init__(
        self, params_schema: dict[str, Any], param_py_types: dict[str, PyType]
    ):
        """Create information about a compute operation

        Args:
            params_schema: map of parameter names to their JSON Schema
                definitions
            param_py_types: map of parameter names to their Python types
        """
        self.params_schema = params_schema
        self.param_py_types = param_py_types

    def make_op(self, function: Callable) -> Callable:
        """Add this information instance to a function

        The supplied function is modified in-place (with the addition of an
        attribute referencing this information instance) and returned.

        Args:
            function: the function to associate with this information

        Returns:
            the same function (with an attribute added)
        """
        setattr(function, _ATTR_NAME_OP_INFO, self)
        return function

    @classmethod
    def get_op_info(cls, op: Callable) -> "OpInfo":
        """Get the information object for a specified function

        Args:
            op: the function for which information is requested

        Returns:
            the function’s associated information object, if present

        Raises:
            ValueError: if there is no associated information object
                (i.e. the function is not an operation)
        """
        op_info = getattr(op, _ATTR_NAME_OP_INFO, None)
        if not isinstance(op_info, OpInfo):
            raise ValueError(
                f"function {op.__name__}() is not" f" registered as an operation"
            )
        return op_info

    @classmethod
    def new_op_info(cls, op: Callable) -> "OpInfo":
        """Create a new operation information object for a function

        The returned information object contains a parameter type dictionary
        and JSON schema created by analysis of the supplied function.

        Args:
            op: a function

        Returns:
            operation information for the supplied function
        """
        members = dict(inspect.getmembers(op))
        annotations = members.get("__annotations__")
        code = members.get("__code__")
        param_py_types = {}
        params_schema = {}
        if code:
            args = inspect.getargs(code)
            required_param_names = set(args.args or []).union(set(args.varargs or []))
            optional_param_names = set(args.varkw or [])
            all_param_names = required_param_names.union(optional_param_names)
            if all_param_names:
                properties = {}
                for param_name in all_param_names:
                    py_type = annotations.get(param_name)
                    if py_type is not None:
                        param_py_types[param_name] = py_type
                    if inspect.isclass(py_type):
                        if issubclass(py_type, (xr.Dataset, MultiLevelDataset)):
                            # extract function from get_effective_parameters
                            param_schema = {
                                "type": "string",
                                "title": "Dataset identifier",
                            }
                        else:
                            raise ValueError(
                                f"Illegal operation parameter class {py_type}."
                                f" Classes must be subclasses of Dataset or "
                                f"MultiLevelDataset."
                            )
                    elif py_type is not None:
                        if not OpInfo._is_valid_parameter_type(py_type):
                            raise ValueError(
                                f"Illegal operation parameter type {py_type}."
                            )
                        param_schema = build_json_schema(py_type).to_dict()
                    else:
                        param_schema = {}
                    properties[param_name] = param_schema
                params_schema = {
                    "type": "object",
                    "properties": properties,
                    "required": list(required_param_names),
                    "additionalProperties": False,
                }
            else:
                params_schema.update(
                    {"type": ["null", "object"], "additionalProperties": False}
                )

        return OpInfo(params_schema, param_py_types)

    @staticmethod
    def _is_valid_parameter_type(py_type) -> bool:
        """Raise an exception if the supplied type is not valid for operations

        "Valid" means that it is composed only of sequences, mappings,
        primitives, and combinations thereof.
        """
        pass
        origin = typing.get_origin(py_type)
        args = typing.get_args(py_type)
        if py_type in [int, float, str]:
            return True
        elif origin in [dict, collections.abc.Mapping]:
            return args[0] is str and OpInfo._is_valid_parameter_type(args[1])
        elif origin in [list, tuple, collections.abc.Sequence]:
            return all(map(OpInfo._is_valid_parameter_type, args))

    @property
    def effective_param_py_types(self) -> dict[str, PyType]:
        """Return effective Python types for the operation’s parameters

        For any parameter which has a Python type annotation, that type will
        be set as the value in the returned mapping. If a parameter has no
        Python type annotation but a primitive type is defined in the
        operation’s JSON schema, the corresponding Python type will be used
        as the value in the returned type dictionary. Otherwise `None` will
        be used.

        Returns:
            a dictionary mapping operation parameter names to their
            effective Python types
        """
        py_types = self.param_py_types.copy()
        for param_name, param_schema in self.param_schemas.items():
            py_type = py_types.get(param_name)
            if py_type is None:
                json_type = param_schema.get("type")
                # TODO Document requirement that operations must have type
                #  annotations
                # TODO check whether matumosho can do the inverse transform
                #  for us
                py_type = _PRIMITIVE_JSON_TO_PY_TYPES.get(json_type)
                if py_type is not None:
                    py_types[param_name] = py_type
        return py_types

    @property
    def param_schemas(self) -> dict[str, Any]:
        """Returns:
        a mapping of parameter names to their JSON schemas
        """
        return self.params_schema.get("properties", {})

    def set_param_schemas(self, schemas: dict[str, Any]):
        """Set JSON schemas for operation parameters

        Args:
            schemas: a mapping of parameter names to their JSON schemas
        """
        self.params_schema["properties"] = schemas

    def update_params_schema(self, schema: dict[str, Any]):
        """Update the JSON schema for the whole parameters dictionary

        Update the existing parameter dictionary schema with the supplied
        dictionary. The supplied dictionary should be a valid JSON Schema,
        with any schemas for the individual parameters in a sub-dictionary
        under the `properties` key.

        Args:
            schema: the schema with which to update the parameters
                dictionary schema
        """
        self.params_schema.update(schema)

    def update_param_schema(self, param_name: str, schema: dict[str, Any]):
        """Update the JSON Schema for a single parameter

        Args:
            param_name: name of the parameter
            schema: schema with which to update the parameter’s
                current schema
        """
        param_schemas = self.param_schemas
        param_schema = param_schemas.get(param_name, {})
        param_schema.update(schema)
        # next line needed in case default value was taken in .get above
        param_schemas[param_name] = param_schema
        self.set_param_schemas(param_schemas)

    def get_param_py_type(self, param_name: str) -> PyType:
        """Get the Python type for a parameter

        Args:
            param_name: name of parameter

        Returns:
            Python type of specified parameter
        """
        return self.param_py_types[param_name]

    def set_param_py_type(self, param_name: str, py_type: PyType):
        """Set the Python type for a parameter

        Args:
            param_name: name of parameter
            py_type: Python type of specified parameter
        """
        self.param_py_types[param_name] = py_type
