# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import json
from typing import Any, Dict, List, Union

import numpy as np

JsonArray = list["JsonValue"]
JsonObject = dict[str, "JsonValue"]
JsonValue = Union[None, bool, int, float, str, JsonArray, JsonObject]


class NumpyJSONEncoder(json.JSONEncoder):
    """A JSON encoder that converts numpy-like
    scalars into corresponding serializable Python objects.
    """

    def default(self, obj: Any) -> JsonValue:
        converted_obj = _convert_default(obj)
        if converted_obj is not obj:
            return converted_obj
        return json.JSONEncoder.default(self, obj)


_PRIMITIVE_JSON_TYPES = {type(None), bool, int, float, str}


def to_json_value(obj: Any) -> JsonValue:
    """Convert *obj* into a JSON-serializable object.

    Args:
        obj: A Python object.

    Returns: A JSON-serializable version of *obj*, or *obj*
        if *obj* is already JSON-serializable.

    Raises:
        TypeError: If *obj* cannot be made JSON-serializable
    """
    if obj is None:
        return None

    converted_obj = _convert_default(obj)
    if converted_obj is not obj:
        return converted_obj

    obj_type = type(obj)

    if obj_type in _PRIMITIVE_JSON_TYPES:
        return obj

    for t in _PRIMITIVE_JSON_TYPES:
        if isinstance(obj, t):
            return t(obj)

    if obj_type is dict:
        converted_obj = {_key(k): to_json_value(v) for k, v in obj.items()}
        if any(converted_obj[k] is not obj[k] for k in obj.keys()):
            return converted_obj
        else:
            return obj

    if obj_type is list:
        converted_obj = [to_json_value(item) for item in obj]
        if any(o1 is not o2 for o1, o2 in zip(converted_obj, obj)):
            return converted_obj
        else:
            return obj

    try:
        return {_key(k): to_json_value(v) for k, v in obj.items()}
    except AttributeError:
        try:
            return [to_json_value(item) for item in obj]
        except TypeError:
            # Same as json.JSONEncoder.default(self, obj)
            raise TypeError(
                f"Object of type {obj.__class__.__name__} is not JSON serializable"
            )


def _key(key: Any) -> str:
    if not isinstance(key, str):
        raise TypeError(
            f"Property names of JSON objects must be strings,"
            f" but got {key.__class__.__name__}"
        )
    return key


def _convert_default(obj: Any) -> Any:
    if hasattr(obj, "dtype") and hasattr(obj, "ndim"):
        if obj.ndim == 0:
            if np.issubdtype(obj.dtype, bool):
                return bool(obj)
            elif np.issubdtype(obj.dtype, np.integer):
                return int(obj)
            elif np.issubdtype(obj.dtype, np.floating):
                return float(obj)
            elif np.issubdtype(obj.dtype, np.datetime64):
                # np.datetime_as_string returns a np._str object;
                # the str() then returns a str object.
                return str(np.datetime_as_string(obj, timezone="UTC", unit="s"))
            elif np.issubdtype(obj.dtype, np.str):
                return str(obj)
        else:
            return [_convert_default(item) for item in obj]
    # We may handle other non-JSON-serializable datatypes here
    return obj
