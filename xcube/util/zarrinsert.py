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

from xcube.util.timecoord import get_time_in_days_since_1970


def check_append_or_insert(time_range, output_path):
    t1, t2 = time_range
    if t1 != t2:
        t_center = (t1 + t2) / 2
    else:
        t_center = t1
    ds = xr.open_zarr(output_path)
    if np.greater(t_center, get_time_in_days_since_1970(ds.time.values[-1])):
        # append modus
        return True
    else:
        return check_if_unique(t_center, output_path)


def check_if_unique(src_time, dst_path):
    """Check if to be added time stamp is unique """
    ds = xr.open_zarr(dst_path)
    mask = np.equal(get_time_in_days_since_1970(ds.time.values[:]), src_time)
    if not mask.any():
        return False
    else:
        print("Timestamp of input file aleady in destination data set, therefore skipping.")
        return None


def merge_single_zarr_into_destination_zarr(src_path, dst_path):
    """Merging the data for the new time stamp into the existing and remaining zarr directory."""
    ds_single = xr.open_zarr(src_path)
    ds = xr.open_zarr(dst_path)
    src_idx = 0
    new_idx = np.amin(np.where(ds.time[:] > (ds_single.time[src_idx])))
    ds.close()
    ds_single.close()
    # Preparing the source directory with the single time stamp to be ready for merging
    # --> files of variables, excluding "lat" and "lon" need to be renamed
    rename_file(src_path, src_idx, new_idx)
    # Preparing the destination directory to be ready for single time stam to be merged
    # --> files of variables, excluding "lat" and "lon" need to be renamed
    # The renaming needs to happen in reversed order and starting at the index of nearest above value:
    for i in reversed(range(new_idx, ds.time.shape[0])):
        rename_file(dst_path, i, (i + 1))
    # Final step: copy the single time stamp files into the destination zarr and adjusting .zarray to the change.
    copy_into_target(src_path, dst_path, new_idx)


def rename_file(path_to_ds, old_index, new_time_i):
    """Renaming files within the directories according to new time index."""
    ds = xr.open_zarr(path_to_ds)
    for v in ds.variables:
        if (v != 'lat') and (v != 'lon') and (v !='lat_bnds') and (v != 'lon_bnds'):
            path = os.path.join(path_to_ds, v)
            for root, dirs, files in os.walk(path):
                for filename in files:
                    if (str(old_index)) in filename[0] and (v != "time"):
                        parts = filename.split('.', 1)
                        new_name = (str(new_time_i) + '.{}').format(parts[1])
                        if new_name != path:
                            os.rename(os.path.join(path, filename), os.path.join(path, new_name))
                    elif (str(old_index)) in filename[0] and (v == "time"):
                        if str(new_time_i) != path:
                            os.rename(os.path.join(path, filename), os.path.join(path, str(new_time_i)))


def copy_into_target(src_path, dst_path, src_index):
    """Copy the files with the new time stamp into the existing zarr directory."""
    ds = xr.open_zarr(src_path)
    for v in ds.variables:
        if (v != 'lat') and (v != 'lon') and (v != 'lat_bnds') and (v != 'lon_bnds'):
            path = os.path.join(src_path, v)
            for root, dirs, files in os.walk(path):
                for filename in files:
                    if str(src_index) in filename[0]:
                        shutil.copyfile((os.path.join(src_path, v, filename)), (os.path.join(dst_path, v, filename)))
            adjust_zarray(dst_path, v)


def adjust_zarray(dst_path, variable):
    """Changing the shape for time in the .zarray file."""
    with open((os.path.join(dst_path, variable, '.zarray')), 'r') as jsonFile:
        data = json.load(jsonFile)
    t_shape = data["shape"]
    data["shape"][0] = t_shape[0] + 1

    with open((os.path.join(dst_path, variable, '.zarray')), 'w') as jsonFile:
        json.dump(data, jsonFile, indent=4)