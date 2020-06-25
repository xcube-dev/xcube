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
from typing import List

import xarray as xr

from xcube.cli._gen2.genconfig import CubeConfig


# noinspection PyUnusedLocal
from xcube.util.progress import observe_progress


def resample_cube(cube: xr.Dataset,
                  cube_config: CubeConfig):
    # TODO: implement me
    return cube


def resample_and_merge_cubes(cubes: List[xr.Dataset],
                             cube_config: CubeConfig) -> xr.Dataset:
    with observe_progress('Resampling cube(s)', len(cubes) + 1) as progress:
        cubes = []
        for cube in cubes:
            resampled_cube = resample_cube(cube, cube_config)
            cubes.append(resampled_cube)
            progress.worked(1)
        merged_cube = xr.merge(cubes)
        progress.worked(1)
        return merged_cube
