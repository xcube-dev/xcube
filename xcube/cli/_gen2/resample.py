# The MIT License (MIT)
# Copyright (c) 2020 by the xcube development team and contributors
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

from xcube.core.store.param import ParamDescriptor
from xcube.core.store.param import ParamDescriptorSet
from xcube.core.store.param import ParamValues

# Need to be aligned with params in resample_cube(cube, **params)
RESAMPLE_PARAMS = ParamDescriptorSet([
    ParamDescriptor('spatial_crs', dtype=str, default='WGS84', value_set=[None, 'WGS84']),
    ParamDescriptor('spatial_coverage', dtype=str),
    ParamDescriptor('spatial_resolution', dtype=float),
    ParamDescriptor('temporal_coverage', dtype=tuple),
    ParamDescriptor('temporal_resolution', dtype=str),
])


def resample_cube(cube: xr.Dataset,
                  spatial_crs: str = None,
                  spatial_coverage: Tuple[float, float, float, float] = None,
                  spatial_resolution: float = None,
                  temporal_coverage: Tuple[str, str] = None,
                  temporal_resolution: str = None):
    # TODO: implement me
    return cube


def resample_and_merge_cubes(cubes, cube_config: ParamValues) -> xr.Dataset:
    cubes = [resample_cube(cube, **cube_config) for cube in cubes]
    return xr.merge(cubes)
