# The MIT License (MIT)
# Copyright (c) 2018 by the xcube development team and contributors
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
from typing import Any, Dict, Optional

import yaml


def load_metadata_yaml(stream):
    metadata = yaml.load(stream)
    if 'global_attributes' in metadata:
        global_attributes = metadata['global_attributes']
        metadata['global_attributes'] = flatten_dict(global_attributes)
    return metadata


def flatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    result = dict()
    value = _flatten_dict_value(d, result, None, False)
    if value is not result:
        raise ValueError('input must be a mapping object')
    # noinspection PyTypeChecker
    return result


_PRIMITIVE_TYPES = (int, float, str, type(None))


def _flatten_dict_value(value: Any,
                        result: Dict[str, Any],
                        parent_name: Optional[str],
                        concat: bool) -> Any:
    if isinstance(value, datetime.datetime):
        return datetime.datetime.isoformat(value)
    elif isinstance(value, datetime.date):
        return datetime.date.isoformat(value)
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
