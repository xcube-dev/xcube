# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
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

import datetime
import fnmatch
from typing import Any, Dict, Optional, Iterable, Tuple, List

UNDEFINED = object()
PRIMITIVE_TYPES = (int, float, str, type(None))

NameAnyDict = Dict[str, Any]
NameDictPairList = List[Tuple[str, Optional[NameAnyDict]]]


def to_resolved_name_dict_pairs(name_dict_pairs: NameDictPairList, container, keep=False) -> NameDictPairList:
    resolved_pairs = []
    for name, value in name_dict_pairs:
        if '*' in name or '?' in name or '[' in name:
            for resolved_name in container:
                if fnmatch.fnmatch(resolved_name, name):
                    resolved_pairs.append((resolved_name, value))
        elif name in container or keep:
            resolved_pairs.append((name, value))
    return resolved_pairs


def to_name_dict_pairs(iterable: Iterable[Any], default_key=None) -> NameDictPairList:
    return [to_name_dict_pair(item, parent=iterable, default_key=default_key)
            for item in iterable]


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
        raise ValueError(f'name must be a string')

    if value is None:
        return name, None

    try:
        # noinspection PyUnresolvedReferences
        value.items()
    except AttributeError as e:
        if default_key:
            value = {default_key: value}
        else:
            raise ValueError(f'value of {name!r} must be a dictionary') from e

    return name, value


def flatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    result = dict()
    value = _flatten_dict_value(d, result, None, False)
    if value is not result:
        raise ValueError('input must be a mapping object')
    # noinspection PyTypeChecker
    return result


def _flatten_dict_value(value: Any,
                        result: Dict[str, Any],
                        parent_name: Optional[str],
                        concat: bool) -> Any:
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
