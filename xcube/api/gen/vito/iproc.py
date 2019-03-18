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

from typing import Tuple, Optional, Collection

import xarray as xr
from xcube.api.gen.default.iproc import DefaultInputProcessor
from xcube.util.reproject import reproject_crs_to_wgs84, get_projection_wkt

from ..iproc import InputProcessor, register_input_processor


class VitoS2PlusInputProcessor(InputProcessor):
    """
    Input processor for VITO's Sentinel-2 Plus Level-2 NetCDF inputs.
    """

    @property
    def name(self) -> str:
        return 'vito-s2plus-l2'

    @property
    def description(self) -> str:
        return 'VITO Sentinel-2 Plus Level 2 NetCDF inputs'

    @property
    def input_reader(self) -> str:
        return 'netcdf4'

    def get_time_range(self, dataset: xr.Dataset) -> Tuple[float, float]:
        return DefaultInputProcessor().get_time_range(dataset)

    def get_extra_vars(self, dataset: xr.Dataset) -> Optional[Collection[str]]:
        return ["transverse_mercator"]

    def pre_process(self, dataset: xr.Dataset) -> xr.Dataset:
        # TODO (forman): clarify with VITO how to correctly mask the S2+ variables
        return super().pre_process(dataset)

    def process(self,
                dataset: xr.Dataset,
                dst_size: Tuple[int, int],
                dst_region: Tuple[float, float, float, float],
                dst_resampling: str,
                include_non_spatial_vars=False) -> xr.Dataset:
        return reproject_crs_to_wgs84(dataset,
                                      self.get_dataset_crs(dataset),
                                      dst_size,
                                      dst_region,
                                      dst_resampling,
                                      include_non_spatial_vars=include_non_spatial_vars)

    @classmethod
    def get_dataset_crs(cls, dataset: xr.Dataset) -> str:
        proj_params = dataset["transverse_mercator"]

        latitude_of_origin = proj_params.attrs["latitude_of_projection_origin"]
        central_meridian = proj_params.attrs["longitude_of_central_meridian"]
        scale_factor = proj_params.attrs["scale_factor_at_central_meridian"]
        false_easting = proj_params.attrs["false_easting"]
        false_northing = proj_params.attrs["false_northing"]

        return get_projection_wkt("Some S2+ Tile",
                                  "Transverse_Mercator",
                                  latitude_of_origin,
                                  central_meridian,
                                  scale_factor,
                                  false_easting,
                                  false_northing)


def init_plugin():
    """ Register plugin. """
    register_input_processor(VitoS2PlusInputProcessor())
