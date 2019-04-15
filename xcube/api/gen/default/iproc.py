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

from typing import Tuple

import xarray as xr

from xcube.util.constants import CRS_WKT_EPSG_4326
from xcube.util.timecoord import get_time_in_days_since_1970
from ..iproc import XYInputProcessor, register_input_processor, ReprojectionInfo


class DefaultInputProcessor(XYInputProcessor):
    """
    Default input processor that expects input datasets to have the xcube standard format:

    * Have dimensions ``lat``, ``lon``, optionally ``time`` of length 1;
    * have coordinate variables ``lat[lat]``, ``lon[lat]``, ``time[time]`` (opt.), ``time_bnds[time, 2]`` (opt.);
    * have coordinate variables ``lat[lat]``, ``lon[lat]`` as decimal degrees on WGS84 ellipsoid,
      both linearly increasing with same constant delta;
    * have coordinate variable ``time[time]`` representing a date+time values with defined CF "units" attribute;
    * have any data variables of form ``<var>[time, lat, lon]``;
    * have global attribute pairs (``time_coverage_start``, ``time_coverage_end``), or (``start_time``, ``stop_time``)
      if ``time`` coordinate is missing.

    The default input processor can be configured by the following parameters:

    * ``input_reader`` the input reader identifier, default is "netcdf4".

    """

    def __init__(self):
        self._input_reader = 'netcdf4'

    @property
    def name(self) -> str:
        return 'default'

    @property
    def description(self) -> str:
        return 'Single-scene NetCDF/CF inputs in xcube standard format'

    def configure(self, input_reader: str = 'netcdf4'):
        self._input_reader = input_reader

    @property
    def input_reader(self) -> str:
        return self._input_reader

    def pre_process(self, dataset: xr.Dataset) -> xr.Dataset:
        self._validate(dataset)
        if "time" in dataset:
            return dataset.squeeze("time")
        return dataset

    def get_reprojection_info(self, dataset: xr.Dataset) -> ReprojectionInfo:
        return ReprojectionInfo(xy_var_names=('lon', 'lat'),
                                xy_crs=CRS_WKT_EPSG_4326,
                                xy_gcp_step=(max(1, len(dataset.lon) // 4),
                                             max(1, len(dataset.lat) // 4)))

    def get_time_range(self, dataset: xr.Dataset) -> Tuple[float, float]:
        time_coverage_start, time_coverage_end = None, None
        if "time" in dataset:
            time = dataset["time"]
            time_bnds_name = time.attrs.get("bounds", "time_bnds")
            if time_bnds_name in dataset:
                time_bnds = dataset[time_bnds_name]
                if time_bnds.shape == (1, 2):
                    time_coverage_start = str(time_bnds[0][0].data)
                    time_coverage_end = str(time_bnds[0][1].data)
            if time_coverage_start is None or time_coverage_end is None:
                time_coverage_start, time_coverage_end = self.get_time_range_from_attrs(dataset)
            if time_coverage_start is None or time_coverage_end is None:
                if time.shape == (1,):
                    time_coverage_start = str(time[0].data)
                    time_coverage_end = time_coverage_start
        if time_coverage_start is None or time_coverage_end is None:
            time_coverage_start, time_coverage_end = self.get_time_range_from_attrs(dataset)
        if time_coverage_start is None or time_coverage_end is None:
            raise ValueError("invalid input: missing time coverage information in dataset")

        return get_time_in_days_since_1970(time_coverage_start), get_time_in_days_since_1970(time_coverage_end)

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

    def _validate(self, dataset):
        self._check_coordinate_var(dataset, "lon", min_length=2)
        self._check_coordinate_var(dataset, "lat", min_length=2)
        if "time" in dataset.dims:
            self._check_coordinate_var(dataset, "time", max_length=1)
            required_dims = ("time", "lat", "lon")
        else:
            required_dims = ("lat", "lon")
        count = 0
        for var_name in dataset.data_vars:
            var = dataset.data_vars[var_name]
            if var.dims == required_dims:
                count += 1
        if count == 0:
            raise ValueError(f"dataset has no variables with required dimensions {required_dims!r}")

    # noinspection PyMethodMayBeStatic
    def _check_coordinate_var(self, dataset: xr.Dataset, coord_var_name: str,
                              min_length: int = None, max_length: int = None):
        if coord_var_name not in dataset.coords:
            raise ValueError(f'missing coordinate variable "{coord_var_name}"')
        coord_var = dataset.coords[coord_var_name]
        if len(coord_var.shape) != 1:
            raise ValueError('coordinate variable "lon" must be 1D')
        coord_var_bnds_name = coord_var.attrs.get("bounds", coord_var_name + "_bnds")
        if coord_var_bnds_name in dataset:
            coord_bnds_var = dataset[coord_var_bnds_name]
            expected_shape = (len(coord_var), 2)
            if coord_bnds_var.shape != expected_shape:
                raise ValueError(f'coordinate bounds variable "{coord_bnds_var}" must have shape {expected_shape!r}')
        else:
            if min_length is not None and len(coord_var) < min_length:
                raise ValueError(f'coordinate variable "{coord_var_name}" must have at least {min_length} value(s)')
            if max_length is not None and len(coord_var) > max_length:
                raise ValueError(f'coordinate variable "{coord_var_name}" must have no more than {max_length} value(s)')


def init_plugin():
    register_input_processor(DefaultInputProcessor())
