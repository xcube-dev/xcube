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
import inspect
import json
from typing import Any, Dict

import xarray as xr

from xcube.util.config import NameDictPairList


def update_var_props(dataset: xr.Dataset,
                     var_name_to_props: NameDictPairList) -> xr.Dataset:
    """
    Update the variables of given *dataset* with the name to properties mapping given by *var_name_to_props*.
    If a property dictionary contains a property "name" that variables weill be renamed to the value
    of the  "name" property. Other properties are used to update the variable's attributes.

    :param dataset: A dataset.
    :param var_name_to_props: List of tuples of the form (variable name, properties dictionary).
    :return: A new dataset.
    """
    if not var_name_to_props:
        return dataset

    var_name_attrs = dict()
    var_renamings = dict()
    new_var_names = set()

    # noinspection PyUnusedLocal,PyShadowingNames
    for var_name, var_props in var_name_to_props:
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


def update_global_attrs(dataset: xr.Dataset, update_mode: str = None, output_metadata: Dict[str, Any] = None,
                        locals: Dict[str, Any] = None) -> xr.Dataset:
    """
    Update the global CF/THREDDS attributes of given *dataset*.

    :param update_mode:
    :param locals:
    :param dataset: The dataset.
    :param output_metadata: Extra metadata.
    :return: A new dataset.
    """
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

    args, _, _, values = inspect.getargvalues(locals)
    if update_mode == 'create':
        cube_gen_param = []
        for value in values:
            if 'input' in value and value != 'input_dataset' or 'output' in value  and value != 'output_metadata':
                cube_gen_param.append((json.dumps({value: str(values[value])})))

        str_cube_gen_param = "\n".join(cube_gen_param).replace("'", "").replace("{", "").replace("}", "").replace(
            '"', '')
        dataset.attrs[
            'history'] = f"""[{datetime.datetime.now().isoformat()}] {update_mode} with \n{str_cube_gen_param}"""
    else:
        cube_gen_param = json.dumps({'input_file': values['input_file']})
        str_cube_gen_param = cube_gen_param.replace("'", "").replace("{", "").replace("}", "").replace('"', '')
        dataset.attrs['history'].update({
            'history': f"""[{datetime.datetime.now().isoformat()}] {update_mode} {str_cube_gen_param}"""})
    # TODO: Question - should we maybe change this to utcnow() ?
    dataset.attrs['date_modified'] = datetime.datetime.now().isoformat()

    return dataset
