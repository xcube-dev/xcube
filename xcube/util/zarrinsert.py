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

import json
import numpy as np
import os
import shutil
import xarray as xr

from typing import Tuple

from xcube.util.timecoord import get_time_in_days_since_1970


def check_append_or_insert(time_range: Tuple[float, float], output_path: str) -> bool:
    t1, t2 = time_range
    if t1 != t2:
        t_center = (t1 + t2) / 2
    else:
        t_center = t1
    with xr.open_zarr(output_path) as ds:
        return np.greater(t_center, get_time_in_days_since_1970(ds.time.values[-1]))


def insert_input_file_into_output_path(input_path: str, output_path: str) -> bool:
    """Merging the data for the new time stamp into the existing and remaining zarr directory."""
    insert = _check_if_insert_into_output_path(input_path, output_path)
    if insert:
        with xr.open_zarr(input_path) as input_ds, xr.open_zarr(output_path) as output_ds:
            input_time_idx = 0
            new_time_idx = np.amin(np.where(output_ds.time[:] > (input_ds.time.values[input_time_idx])))
        # Preparing the source directory with the single time stamp to be ready for merging
        # --> files of variables, excluding "lat" and "lon" need to be renamed
            _rename_file(input_path, input_time_idx, new_time_idx)
            # Preparing the destination directory to be ready for single time stam to be merged
            # --> files of variables, excluding "lat" and "lon" need to be renamed
            # The renaming needs to happen in reversed order and starting at the index of nearest above value:
            for i in reversed(range(new_time_idx, output_ds.time.shape[0])):
                _rename_file(output_path, i, (i + 1))
            # Final step: copy the single time stamp files into the destination zarr and adjusting .zarray to the change.
            _copy_into_output_path(input_path, output_path, new_time_idx)
            return True
    else:
        return False


def _check_if_insert_into_output_path(input_path: str, output_path: str) -> bool:
    """Check if to be added time stamp is unique """
    with xr.open_zarr(input_path) as input_ds, xr.open_zarr(output_path) as output_ds:
        mask = np.equal(output_ds.time.values, input_ds.time.values)
        return not mask.any()


def _rename_file(path: str, old_time_idx: int, new_time_idx: int):
    """Renaming files within the directories according to new time index."""
    with xr.open_zarr(path) as ds:
        variables = ds.variables
    for v in variables:
        if (v != 'lat') and (v != 'lon') and (v != 'lat_bnds') and (v != 'lon_bnds'):
            v_path = os.path.join(path, v)
            for root, dirs, files in os.walk(v_path):
                for filename in files:
                    parts1 = filename.split('.', 1)[0]
                    if parts1 == (str(old_time_idx)) and (v != "time"):
                        parts2 = filename.split('.', 1)[1]
                        new_name = (str(new_time_idx) + '.{}').format(parts2)
                        os.rename(os.path.join(v_path, filename), os.path.join(v_path, new_name))
                    elif parts1 == (str(old_time_idx)) and (v == "time"):
                        os.rename(os.path.join(v_path, filename), os.path.join(v_path, str(new_time_idx)))


def _copy_into_output_path(input_path: str, output_path: str, input_time_idx: int):
    """Copy the files with the new time stamp into the existing zarr directory."""
    with xr.open_zarr(input_path) as input_ds:
        variables = input_ds.variables
    for variable in variables:
        if (variable != 'lat') and (variable != 'lon') and (variable != 'lat_bnds') and (variable != 'lon_bnds'):
            v_path = os.path.join(input_path, variable)
            for root, dirs, files in os.walk(v_path):
                for filename in files:
                    parts1 = filename.split('.', 1)[0]
                    if parts1 == str(input_time_idx):
                        shutil.copyfile((os.path.join(input_path, variable, filename)), (os.path.join(output_path, variable, filename)))
            _adjust_zarray(output_path, variable)


def _adjust_zarray(output_path: str, variable: str):
    """Changing the shape for time in the .zarray file."""
    with open((os.path.join(output_path, variable, '.zarray')), 'r') as jsonFile:
        data = json.load(jsonFile)
    t_shape = data["shape"]
    data["shape"][0] = t_shape[0] + 1

    with open((os.path.join(output_path, variable, '.zarray')), 'w') as jsonFile:
        json.dump(data, jsonFile, indent=4)
