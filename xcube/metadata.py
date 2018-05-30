import datetime
from typing import Any, Dict, Optional

import yaml


def load_yaml(stream):
    hierchical_metadata = yaml.load(stream)
    return flatten_dict(hierchical_metadata)


def flatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    result = dict()
    value = _flatten_dict_value(d, result, None, False)
    if value is not result:
        raise ValueError('input must be a mapping object')
    return result


_PRIMITIVE_TYPES = (int, float, str, type(None))


def _flatten_dict_value(value: Any,
                        result: Dict[str, Any],
                        parent_name: Optional[str],
                        concat: bool) -> Any:
    if isinstance(value, datetime.date):
        return datetime.date.isoformat(value)
    elif isinstance(value, datetime.datetime):
        return datetime.datetime.isoformat(value)
    elif isinstance(value, _PRIMITIVE_TYPES):
        return value

    try:
        items = value.items()
    except AttributeError:
        items = None

    if items:
        for k, v in items:
            if not isinstance(k, str):
                raise ValueError('all keys must be strings')
            v = _flatten_dict_value(v, result, k, False)
            if v is not result:
                name = k if parent_name is None else parent_name + '_' + k
                if concat and name in result:
                    result[name] = f'{result[name]}, {v}'
                else:
                    result[name] = v

    for e in value:
        _flatten_dict_value(e, result, parent_name, True)

    return result
