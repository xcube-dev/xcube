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
import functools
import math
from typing import Tuple, Any, Dict

import numpy as np
import pandas as pd
import xarray as xr

from .config import NameDictPairList, to_resolved_name_dict_pairs
from .expression import compute_array_expr
from .maskset import MaskSet

REF_DATETIME_STR = '1970-01-01 00:00:00'
REF_DATETIME = pd.to_datetime(REF_DATETIME_STR)
DATETIME_UNITS = f'days since {REF_DATETIME_STR}'
DATETIME_CALENDAR = 'gregorian'
SECONDS_PER_DAY = 24 * 60 * 60
MICROSECONDS_PER_DAY = 1000 * 1000 * SECONDS_PER_DAY


def compute_dataset(dataset: xr.Dataset,
                    processed_variables: NameDictPairList = None,
                    errors: str = 'raise') -> xr.Dataset:
    """
    Compute a dataset from another dataset and return it.

    :param dataset: xarray dataset.
    :param processed_variables: Optional list of variables that will be loaded or computed in the order given.
           Each variable is either identified by name or by a name to variable attributes mapping.
    :param errors: How to deal with errors while evaluating expressions.
           May be be one of "raise", "warn", or "ignore".
    :return: new dataset with computed variables
    """

    if processed_variables:
        processed_variables = to_resolved_name_dict_pairs(processed_variables, dataset, keep=True)
    else:
        var_names = list(dataset.data_vars)
        var_names = sorted(var_names, key=functools.partial(get_var_sort_key, dataset))
        processed_variables = [(var_name, None) for var_name in var_names]

    # Initialize namespace with some constants and modules
    namespace = dict(NaN=np.nan, PI=math.pi, np=np, xr=xr)
    # Now add all mask sets and variables
    for var_name in dataset.data_vars:
        var = dataset[var_name]
        if MaskSet.is_flag_var(var):
            namespace[var_name] = MaskSet(var)
        else:
            namespace[var_name] = var

    for var_name, var_props in processed_variables:
        if var_name in dataset.data_vars:
            # Existing variable
            var = dataset[var_name]
            if var_props:
                var_props_temp = var_props
                var_props = dict(var.attrs)
                var_props.update(var_props_temp)
            else:
                var_props = dict(var.attrs)
        else:
            # Computed variable
            var = None
            if var_props is None:
                var_props = dict()

        expression = var_props.get('expression')
        if expression:
            # Compute new variable
            computed_array = compute_array_expr(expression,
                                                namespace=namespace,
                                                result_name=f'{var_name!r}',
                                                errors=errors)
            if computed_array is not None:
                if hasattr(computed_array, 'attrs'):
                    var = computed_array
                    var.attrs.update(var_props)
                namespace[var_name] = computed_array

        valid_pixel_expression = var_props.get('valid_pixel_expression')
        if valid_pixel_expression:
            # Compute new mask for existing variable
            if var is None:
                raise ValueError(f'undefined variable {var_name!r}')
            valid_mask = compute_array_expr(valid_pixel_expression,
                                            namespace=namespace,
                                            result_name=f'valid mask for {var_name!r}',
                                            errors=errors)
            if valid_mask is not None:
                masked_var = var.where(valid_mask)
                if hasattr(masked_var, 'attrs'):
                    masked_var.attrs.update(var_props)
                namespace[var_name] = masked_var

    computed_dataset = dataset.copy()
    for name, value in namespace.items():
        if isinstance(value, xr.DataArray):
            computed_dataset[name] = value

    return computed_dataset


def select_variables(dataset: xr.Dataset, var_names) -> xr.Dataset:
    if var_names is None:
        return dataset
    dropped_variables = set(dataset.data_vars.keys()).difference(var_names)
    if not dropped_variables:
        return dataset
    return dataset.drop(dropped_variables)


def update_variable_props(dataset: xr.Dataset,
                          var_name_props_pair_list: NameDictPairList) -> xr.Dataset:
    if not var_name_props_pair_list:
        return dataset

    var_name_attrs = dict()
    var_renamings = dict()
    new_var_names = set()

    # noinspection PyUnusedLocal,PyShadowingNames
    for var_name, var_props in var_name_props_pair_list:
        if not var_props:
            continue
        # noinspection PyShadowingNames
        var_attrs = dict(var_props)
        if 'name' in var_attrs:
            new_var_name = var_attrs.pop('name')
            if new_var_name in new_var_names:
                raise ValueError(f'variable {var_name!r} cannot be renamed into {new_var_name!r} '
                                 'because the name is already in use')
            new_var_names.add(new_var_name)
            var_attrs['original_name'] = var_name
            var_renamings[var_name] = new_var_name
            var_name = new_var_name
        var_name_attrs[var_name] = var_attrs

    if var_renamings:
        dataset = dataset.rename(var_renamings)
    elif var_name_attrs:
        dataset = dataset.copy()

    if var_name_attrs:
        for var_name, var_attrs in var_name_attrs.items():
            var = dataset[var_name]
            var.attrs.update(var_attrs)

    return dataset


def get_var_sort_key(dataset: xr.Dataset, var_name: str):
    # noinspection SpellCheckingInspection
    attrs = dataset[var_name].attrs
    a1 = attrs.get('expression')
    a2 = attrs.get('valid_pixel_expression')
    v1 = 10 * len(a1) if a1 is not None else 0
    v2 = 100 * len(a2) if a2 is not None else 0
    return v1 + v2


def add_time_coords(dataset: xr.Dataset, time_range: Tuple[float, float]) -> xr.Dataset:
    t1, t2 = time_range
    if t1 != t2:
        t_center = (t1 + t2) / 2
    else:
        t_center = t1
    dataset = dataset.expand_dims('time')
    dataset = dataset.assign_coords(time=(['time'], [t_center]))
    time_var = dataset.coords['time']
    time_var.attrs['long_name'] = 'time'
    time_var.attrs['standard_name'] = 'time'
    time_var.attrs['units'] = DATETIME_UNITS
    time_var.attrs['calendar'] = DATETIME_CALENDAR
    time_var.encoding['units'] = DATETIME_UNITS
    time_var.encoding['calendar'] = DATETIME_CALENDAR
    if t1 != t2:
        time_var.attrs['bounds'] = 'time_bnds'
        dataset = dataset.assign_coords(time_bnds=(['time', 'bnds'], [[t1, t2]]))
        time_bnds_var = dataset.coords['time_bnds']
        time_bnds_var.attrs['long_name'] = 'time'
        time_bnds_var.attrs['standard_name'] = 'time'
        time_bnds_var.attrs['units'] = DATETIME_UNITS
        time_bnds_var.attrs['calendar'] = DATETIME_CALENDAR
        time_bnds_var.encoding['units'] = DATETIME_UNITS
        time_bnds_var.encoding['calendar'] = DATETIME_CALENDAR
    return dataset


def get_time_in_days_since_1970(time_str: str, pattern=None) -> float:
    datetime = pd.to_datetime(time_str, format=pattern, infer_datetime_format=True)
    timedelta = datetime - REF_DATETIME
    return timedelta.days + timedelta.seconds / SECONDS_PER_DAY + timedelta.microseconds / MICROSECONDS_PER_DAY


def update_global_attributes(dataset: xr.Dataset, output_metadata: Dict[str, Any] = None) -> xr.Dataset:
    dataset = dataset.copy()
    if output_metadata:
        dataset.attrs.update(output_metadata)

    data = [('lon', 'lon_bnds', 'degrees_east', ('geospatial_lon_min', 'geospatial_lon_max',
                                                 'geospatial_lon_units', 'geospatial_lon_resolution'), float),
            ('lat', 'lat_bnds', 'degrees_north', ('geospatial_lat_min', 'geospatial_lat_max',
                                                  'geospatial_lat_units', 'geospatial_lat_resolution'), float),
            ('time', 'time_bnds', None, ('time_coverage_start', 'time_coverage_end',
                                         None, None), str)]
    for coord_name, coord_bnds_name, coord_units, coord_attr_names, cast in data:
        coord_min_attr_name, coord_max_attr_name, coord_units_attr_name, coord_res_attr_name = coord_attr_names
        if coord_min_attr_name not in dataset.attrs or coord_max_attr_name not in dataset.attrs:
            coord = None
            coord_bnds = None
            coord_res = None
            if coord_name in dataset:
                coord = dataset[coord_name]
                coord_bnds_name = coord.attrs.get('bounds', coord_bnds_name)
            if coord_bnds_name in dataset:
                coord_bnds = dataset[coord_bnds_name]
            if coord_bnds is not None and coord_bnds.ndim == 2 and coord_bnds.shape[0] > 1 and coord_bnds.shape[1] == 2:
                coord_v1 = coord_bnds[0][0]
                coord_v2 = coord_bnds[-1][1]
                coord_res = (coord_v2 - coord_v1) / coord_bnds.shape[0]
                coord_res = float(coord_res.values)
                coord_min, coord_max = (coord_v1, coord_v2) if coord_res > 0 else (coord_v2, coord_v1)
                dataset.attrs[coord_min_attr_name] = cast(coord_min.values)
                dataset.attrs[coord_max_attr_name] = cast(coord_max.values)
            elif coord is not None and coord.ndim == 1 and coord.shape[0] > 1:
                coord_v1 = coord[0]
                coord_v2 = coord[-1]
                coord_res = (coord_v2 - coord_v1) / (coord.shape[0] - 1)
                coord_v1 -= coord_res / 2
                coord_v2 += coord_res / 2
                coord_res = float(coord_res.values)
                coord_min, coord_max = (coord_v1, coord_v2) if coord_res > 0 else (coord_v2, coord_v1)
                dataset.attrs[coord_min_attr_name] = cast(coord_min.values)
                dataset.attrs[coord_max_attr_name] = cast(coord_max.values)
            if coord_units_attr_name is not None and coord_units is not None:
                dataset.attrs[coord_units_attr_name] = coord_units
            if coord_res_attr_name is not None and coord_res is not None:
                dataset.attrs[coord_res_attr_name] = coord_res if coord_res > 0 else -coord_res

    dataset.attrs['date_modified'] = datetime.datetime.now().isoformat()

    return dataset
