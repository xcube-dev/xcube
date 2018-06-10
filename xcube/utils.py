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
from typing import Set

import xarray as xr


def select_variables(dataset: xr.Dataset, var_name_patterns: Set[str]) -> xr.Dataset:
    if not var_name_patterns:
        return dataset

    var_names = set()
    for var_name_pattern in var_name_patterns:
        if '*' in var_name_pattern or '?' in var_name_pattern or '[' in var_name_pattern:
            for var_name in dataset.data_vars:
                if fnmatch.fnmatch(var_name, var_name_pattern):
                    var_names.add(var_name)
        elif var_name_pattern in dataset.data_vars:
            var_names.add(var_name_pattern)

    dropped_variables = set(dataset.data_vars.keys()).difference(var_names)
    if not dropped_variables:
        return dataset

    return dataset.drop(dropped_variables)

