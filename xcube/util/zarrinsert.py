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
from typing import Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from xcube.util.timecoord import get_time_in_days_since_1970


def check_append_or_insert(time_range, output_path):
    t1, t2 = time_range
    if t1 != t2:
        t_center = (t1 + t2) / 2
    else:
        t_center = t1
    ds = xr.open_zarr(output_path)
    if np.greater(t_center, get_time_in_days_since_1970(ds.time[-1]).all()):
        # append modus
        print("Append modus is chosen.")
        return True
    else:
        check_if_unique(t_center, output_path)
#         print("Merge modus is chosen.")

def check_if_unique(src_time, dst_path):
    print('check unique')
    """Check if to be added time stamp is unique """
    ds = xr.open_zarr(dst_path)
    mask = np.equal(get_time_in_days_since_1970(ds.time), src_time)
    if not mask.all():
        merge_single_zarr_into_destination_zarr(src_time, dst_path, src_idx)
    else:
        print("All timestamps to be merged are aleady in destination data set, and are skipped.")