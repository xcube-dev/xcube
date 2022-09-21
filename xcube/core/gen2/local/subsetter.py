# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
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
import math

import pyproj
import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.core.select import select_spatial_subset
from xcube.core.select import select_temporal_subset
from xcube.core.select import select_variables_subset
from .transformer import CubeTransformer
from .transformer import TransformedCube
from ..config import CubeConfig


class CubeSubsetter(CubeTransformer):

    def transform_cube(self,
                       cube: xr.Dataset,
                       gm: GridMapping,
                       cube_config: CubeConfig) -> TransformedCube:

        desired_var_names = cube_config.variable_names
        if desired_var_names:
            cube = select_variables_subset(cube,
                                           var_names=desired_var_names)
            cube_config = cube_config.drop_props('variable_names')

        desired_bbox = cube_config.bbox
        if desired_bbox is not None:
            # Find out whether its possible to make a spatial subset
            # without resampling. First, grid mapping must be regular.
            can_do_spatial_subset = False
            if gm.is_regular:
                can_do_spatial_subset = True
                # Current spatial resolution must be the
                # desired spatial resolution, otherwise spatial resampling
                # is required later, which will include the desired
                # subsetting.
                desired_res = cube_config.spatial_res
                if desired_res is not None \
                        and not (math.isclose(gm.x_res, desired_res)
                                 and math.isclose(gm.y_res, desired_res)):
                    can_do_spatial_subset = False
                if can_do_spatial_subset:
                    # Finally, the desired CRS must be equal to the current
                    # one, or they must both be geographic.
                    desired_crs = cube_config.crs
                    if desired_crs:
                        desired_crs = pyproj.CRS.from_string(desired_crs)
                        if desired_crs != gm.crs \
                                and not (desired_crs.is_geographic
                                         and gm.crs.is_geographic):
                            can_do_spatial_subset = False
            if can_do_spatial_subset:
                cube = select_spatial_subset(cube,
                                             xy_bbox=desired_bbox)
                # Now that we have a new cube subset, we must adjust
                # its grid mapping.
                gm = GridMapping.from_dataset(
                    cube,
                    crs=gm.crs,
                )
                # Consume spatial properties
                cube_config = cube_config.drop_props(['bbox',
                                                      'spatial_res',
                                                      'crs'])

        desired_time_range = cube_config.time_range
        if desired_time_range:
            cube = select_temporal_subset(cube,
                                          time_range=desired_time_range)
            cube_config = cube_config.drop_props('time_range')

        return cube, gm, cube_config
