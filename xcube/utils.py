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

import fnmatch
from typing import Any, Iterable, Callable, Optional, Dict, Set

import xarray as xr


def iter_variables(dataset: xr.Dataset,
                   var_name_patterns: Iterable[Any],
                   callback: Callable[[xr.Dataset, Optional[Dict], Any, Optional[str]], None],
                   default_prop_name='name') -> None:
    if not var_name_patterns:
        return

    for var_name_pattern in var_name_patterns:
        var_name_pattern, var_props = to_key_dict_pair(var_name_pattern,
                                                       parent=var_name_patterns,
                                                       default_key=default_prop_name)

        if '*' in var_name_pattern or '?' in var_name_pattern or '[' in var_name_pattern:
            for var_name in dataset.data_vars:
                if fnmatch.fnmatch(var_name, var_name_pattern):
                    callback(dataset, var_name, var_props, var_name_pattern)
        elif var_name_pattern in dataset.data_vars:
            callback(dataset, var_name_pattern, var_props, None)


_UNDEFINED = object()


def to_key_dict_pair(key: Any, parent: Any = None, default_key: str = None):
    value = _UNDEFINED

    try:
        # is key of parent?
        value = parent[key]
    except (KeyError, TypeError, ValueError):
        # key = {key: value}?
        try:
            key, value = dict(key).popitem()
        except (TypeError, ValueError, AttributeError, KeyError):
            if not isinstance(key, str):
                # key = (key, value)?
                try:
                    key, value = key
                except (TypeError, ValueError):
                    pass

    if value is _UNDEFINED:
        return key, None

    if default_key:
        try:
            value.items()
        except AttributeError:
            value = {default_key: value}

    return key, value


def get_valid_variable_names(dataset: xr.Dataset, var_name_patterns: Iterable[Any]) -> Optional[Set[str]]:
    if var_name_patterns is None:
        return None

    var_names = set()

    # noinspection PyUnusedLocal
    def callback(ds, var_name, *args):
        var_names.add(var_name)

    iter_variables(dataset, var_name_patterns, callback)

    return var_names


def select_variables(dataset: xr.Dataset, var_name_patterns: Iterable[Any]) -> xr.Dataset:
    var_names = get_valid_variable_names(dataset, var_name_patterns)
    if var_names is None:
        return dataset
    dropped_variables = set(dataset.data_vars.keys()).difference(var_names)
    if not dropped_variables:
        return dataset
    return dataset.drop(dropped_variables)


def update_variable_props(dataset: xr.Dataset,
                          var_name_pattern_props: Iterable[Any]) -> xr.Dataset:
    var_name_attrs = dict()
    var_renamings = dict()

    # noinspection PyUnusedLocal,PyShadowingNames
    def callback(ds, var_name, var_props, var_name_pattern):
        if not var_props:
            return
        # noinspection PyShadowingNames
        var_attrs = dict(var_props)
        if 'name' in var_attrs:
            var_name_new = var_attrs.pop('name')
            if var_name_pattern is not None:
                raise ValueError(f'variable pattern {var_name_pattern!r} cannot be renamed into {var_name_new!r}')
            var_attrs['original_name'] = var_name
            var_renamings[var_name] = var_name_new
            var_name = var_name_new
        var_name_attrs[var_name] = var_attrs

    iter_variables(dataset, var_name_pattern_props, callback)

    if var_renamings:
        dataset = dataset.rename(var_renamings)
    elif var_name_attrs:
        dataset = dataset.copy()

    if var_name_attrs:
        for var_name, var_attrs in var_name_attrs.items():
            var = dataset[var_name]
            var.attrs.update(var_attrs)

    return dataset
