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

from abc import ABCMeta
from typing import Tuple

import xarray as xr

from ..inputprocessor import InputProcessor, register_input_processor
from ...dsutil import get_time_in_days_since_1970


class DefaultInputProcessor(InputProcessor, metaclass=ABCMeta):
    """
    Default input processor that expects input datasets to have the xcube standard format:

    * Have dimensions ``lat``, ``lon``, optionally ``time`` of length 1;
    * have coordinate variables ``lat[lat]``, ``lon[lat]``, ``time[time]`` (opt.), ``time_bnds[time, 2]`` (opt.);
    * have any data variables of form ``<var>[time, lat, lon]``;
    * have global attribute pairs ``time_coverage_start``, ``time_coverage_end`` if ``time`` coordinate is missing.

    """

    @property
    def name(self) -> str:
        return 'default'

    @property
    def description(self) -> str:
        return 'Single-scene NetCDF inputs in xcube standard format'

    @property
    def input_reader(self) -> str:
        return 'netcdf4'

    def pre_process(self, dataset: xr.Dataset) -> xr.Dataset:
        has_time_coords = "time" in dataset.dims
        var_names_to_drop = []
        for var_name in dataset.data_vars:
            var = dataset.data_vars[var_name]
            if has_time_coords:
                use_var = var.dims == ("time", "lat", "lon")
            else:
                use_var = var.dims == ("lat", "lon")
            if not use_var:
                var_names_to_drop.append(var_name)
        return dataset.drop(var_names_to_drop) if var_names_to_drop else dataset

    def get_reprojection_info(self, dataset: xr.Dataset) -> None:
        return None

    def get_time_range(self, dataset: xr.Dataset) -> Tuple[float, float]:
        time_coverage_start, time_coverage_end = None, None
        if "time" in dataset:
            time = dataset["time"]
            time_bnds_name = time.attrs("bounds", "time_bnds")
            if time_bnds_name in dataset:
                time_bnds = dataset[time_bnds_name]
                if time_bnds.shape == (1, 2):
                    time_coverage_start = str(time_bnds[0][0])
                    time_coverage_end = str(time_bnds[0][1])
            if time_coverage_start is None or time_coverage_end is None:
                time_coverage_start, time_coverage_end = self.get_time_range_from_attrs(dataset)
            if time_coverage_start is None or time_coverage_end is None:
                if time.shape == (1,):
                    time_coverage_start = str(time[0])
                    time_coverage_end = time_coverage_start
        if time_coverage_start is None or time_coverage_end is None:
            time_coverage_start, time_coverage_end = self.get_time_range_from_attrs(dataset)
        if time_coverage_start is None or time_coverage_end is None:
            raise ValueError("invalid input: missing time coverage information in dataset")

        return get_time_in_days_since_1970(time_coverage_start), \
               get_time_in_days_since_1970(time_coverage_end)

    @classmethod
    def get_time_range_from_attrs(cls, dataset: xr.Dataset) -> Tuple[str, str]:
        time_start = time_stop = None
        if "time_coverage_start" in dataset.attrs:
            time_start = str(dataset.attrs["time_coverage_start"])
            time_stop = str(dataset.attrs.get("time_coverage_end", time_start))
        elif "time_start" in dataset.attrs:
            time_start = str(dataset.attrs["time_start"])
            time_stop = str(dataset.attrs.get("time_stop", dataset.attrs.get("time_end", time_start)))
        elif "start_time" in dataset.attrs:
            time_start = str(dataset.attrs["start_time"])
            time_stop = str(dataset.attrs.get("stop_time", dataset.attrs.get("end_time", time_start)))
        return time_start, time_stop


def init_plugin():
    register_input_processor(DefaultInputProcessor())
