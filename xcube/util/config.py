# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import datetime
import fnmatch
import io
import json
import os
import os.path
import string
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple, Type

import fsspec
import yaml

from xcube.constants import LOG

PRIMITIVE_TYPES = (int, float, str, type(None))

NameAnyDict = dict[str, Any]
NameDictPairList = list[tuple[str, Optional[NameAnyDict]]]


def to_resolved_name_dict_pairs(
    name_dict_pairs: NameDictPairList, container, keep=False
) -> NameDictPairList:
    resolved_pairs = []
    for name, value in name_dict_pairs:
        if "*" in name or "?" in name or "[" in name:
            for resolved_name in container:
                if fnmatch.fnmatch(resolved_name, name):
                    resolved_pairs.append((resolved_name, value))
        elif name in container or keep:
            resolved_pairs.append((name, value))
    return resolved_pairs


def to_name_dict_pairs(iterable: Iterable[Any], default_key=None) -> NameDictPairList:
    return [
        to_name_dict_pair(item, parent=iterable, default_key=default_key)
        for item in iterable
    ]


def to_name_dict_pair(name: Any, parent: Any = None, default_key=None):
    value = None

    if isinstance(name, str):
        try:
            # is key of parent?
            value = parent[name]
        except (KeyError, TypeError, ValueError):
            pass
    else:
        # key = (key, value)?
        try:
            name, value = name
        except (TypeError, ValueError):
            # key = {key: value}?
            try:
                name, value = dict(name).popitem()
            except (TypeError, ValueError, AttributeError, KeyError):
                pass

    if not isinstance(name, str):
        raise ValueError(f"name must be a string")

    if value is None:
        return name, None

    try:
        # noinspection PyUnresolvedReferences
        value.items()
    except AttributeError as e:
        if default_key:
            value = {default_key: value}
        else:
            raise ValueError(f"value of {name!r} must be a dictionary") from e

    return name, value


def flatten_dict(d: dict[str, Any]) -> dict[str, Any]:
    result = dict()
    value = _flatten_dict_value(d, result, None, False)
    if value is not result:
        raise ValueError("input must be a mapping object")
    # noinspection PyTypeChecker
    return result


def _flatten_dict_value(
    value: Any, result: dict[str, Any], parent_name: Optional[str], concat: bool
) -> Any:
    if isinstance(value, datetime.datetime):
        return datetime.datetime.isoformat(value)
    elif isinstance(value, datetime.date):
        return datetime.date.isoformat(value)
    elif isinstance(value, PRIMITIVE_TYPES):
        return value

    try:
        items = value.items()
    except AttributeError:
        items = None

    if items:
        for k, v in items:
            if not isinstance(k, str):
                raise ValueError("all keys must be strings")
            v = _flatten_dict_value(v, result, k, False)
            if v is not result:
                name = k if parent_name is None else parent_name + "_" + k
                if concat and name in result:
                    result[name] = f"{result[name]}, {v}"
                else:
                    result[name] = v

    for e in value:
        _flatten_dict_value(e, result, parent_name, True)

    return result


def merge_config(first_dict: dict, *more_dicts):
    if not more_dicts:
        output_dict = first_dict
    else:
        output_dict = dict(first_dict)
        for d in more_dicts:
            for k, v in d.items():
                if (
                    k in output_dict
                    and isinstance(output_dict[k], dict)
                    and isinstance(v, dict)
                ):
                    v = merge_config(output_dict[k], v)
                output_dict[k] = v
    return output_dict


def load_configs(
    *config_paths: str, exception_type: type[Exception] = ValueError
) -> dict[str, Any]:
    config_dicts = []
    for config_path in config_paths:
        config_dict = load_json_or_yaml_config(
            config_path, exception_type=exception_type
        )
        config_dicts.append(config_dict)
    config = merge_config(*config_dicts)
    return config


def load_json_or_yaml_config(
    config_path: str, exception_type: type[Exception] = ValueError
) -> dict[str, Any]:
    try:
        config_dict = _load_json_or_yaml_config(config_path)
        LOG.info(f"Configuration loaded: {config_path}")
    except FileNotFoundError as e:
        raise exception_type(f"Cannot find configuration {config_path!r}") from e
    except yaml.YAMLError as e:
        raise exception_type(f"YAML in {config_path!r} is invalid: {e}") from e
    except OSError as e:
        raise exception_type(
            f"Cannot load configuration from {config_path!r}: {e}"
        ) from e
    if config_dict is None:
        return {}
    if not isinstance(config_dict, dict):
        raise exception_type(
            f"Invalid configuration format in {config_path!r}: dictionary expected"
        )
    return config_dict


def _load_json_or_yaml_config(config_file: str) -> Any:
    with fsspec.open(config_file, mode="r") as fp:
        file_content = fp.read()
    template = string.Template(file_content)
    file_content = template.safe_substitute(os.environ)
    with io.StringIO(file_content) as fp:
        if config_file.endswith(".json"):
            return json.load(fp)
        else:
            return yaml.safe_load(fp)
