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
import os.path
import shutil
import warnings
from typing import Callable, Type
import xarray as xr
import zarr

from xcube.util.config import load_configs
from xcube.util.optimize import optimize_dataset
from xcube.util.update import update_dataset_attrs

_NO_MANUAL_EDIT = ['geospatial_lon_min', 'geospatial_lon_max', 'geospatial_lon_units', 'geospatial_lon_resolution',
                   'geospatial_lat_min', 'geospatial_lat_max', 'geospatial_lat_units', 'geospatial_lat_resolution',
                   'time_coverage_start', 'time_coverage_end']


def edit_metadata(input_path: str,
                  output_path: str = None,
                  metadata_path: str = None,
                  coords: bool = False,
                  in_place: bool = False,
                  monitor: Callable[..., None] = None,
                  exception_type: Type[Exception] = ValueError):
    """
    Edit the metadata of an xcube dataset.

    Editing the metadata because it may be incorrect, inconsistent or incomplete.
    The metadata attributes should be given by a yaml file with the keywords to be edited.
    The function currently works only for data cubes using ZARR format.

    :param input_path: Path to input dataset with ZARR format.
    :param output_path: Path to output dataset with ZARR format. May contain "{input}" template string,
           which is replaced by the input path's file name without file name extentsion.
    :param metadata_path: Path to the metadata file, which will edit the existing metadata.
    :param coords: Whether to update the metadata about the coordinates.
    :param in_place: Whether to modify the dataset in place.
           If False, a copy is made and *output_path* must be given.
    :param monitor: A progress monitor.
    :param exception_type: Type of exception to be used on value errors.
    """

    if not os.path.isfile(os.path.join(input_path, '.zgroup')):
        raise exception_type('Input path must point to ZARR dataset directory.')

    input_path = os.path.abspath(os.path.normpath(input_path))

    if in_place:
        output_path = input_path
    else:
        if not output_path:
            raise exception_type(f'Output path must be given.')
        if '{input}' in output_path:
            base_name, _ = os.path.splitext(os.path.basename(input_path))
            output_path = output_path.format(input=base_name)
        output_path = os.path.abspath(os.path.normpath(output_path))
        if os.path.exists(output_path):
            raise exception_type(f'Output path already exists.')

    if not in_place:
        shutil.copytree(input_path, output_path)

    if monitor is None:
        # noinspection PyUnusedLocal
        def monitor(*args):
            pass

    cube = zarr.open(output_path)

    if coords:
        with xr.open_zarr(output_path) as ds:
            ds_attrs = update_dataset_attrs(ds, update_existing=False, in_place=True).attrs
        for key in ds_attrs:
            cube.attrs.update({key: ds_attrs[key]})

    if metadata_path:
        new_metadata = load_configs(metadata_path)

        for element in new_metadata:
            if 'output_metadata' in element:
                _edit_keyvalue_in_metadata(cube, new_metadata, element, monitor)
            else:
                if cube.__contains__(element):
                    _edit_keyvalue_in_metadata(cube[element], new_metadata, element, monitor)
                else:
                    warnings.warn(f'The variable "{element}" could not be found in the xcube dataset. '
                                  f'Please check spelling of it.')

    # the metadata attrs of a consolidated xcube dataset may not be changed
    # (https://zarr.readthedocs.io/en/stable/api/convenience.html#zarr.convenience.consolidate_metadata)
    # therefore after changing metadata the xcube dataset needs to be consolidated once more.
    if os.path.exists(os.path.join(output_path, '.zmetadata')):
        optimize_dataset(output_path, in_place=True)


def _edit_keyvalue_in_metadata(cube, new_metadata, element, monitor):
    for key in new_metadata[element].keys():
        if key in _NO_MANUAL_EDIT:
            monitor(f'"{key}" is not updated in the global attributes of the xcube dataset. '
                    f'Please use "--coords" for updating coordinate information.')
        else:
            cube.attrs.update({key: new_metadata[element][key]})
            if 'output_metadata' in element:
                monitor(f'Updated "{key}" in the global attributes of the xcube dataset.')
            else:
                monitor(f'Updated "{key}" in the attributes of "{element}" in the xcube dataset.')
