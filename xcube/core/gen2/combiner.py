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

from typing import Sequence

import xarray as xr

from xcube.core.gen2.config import CubeConfig
from xcube.util.progress import observe_progress
from .processor import CubesProcessor
from .rechunker import CubeRechunker
from .resampler import CubeResampler


class CubesCombiner(CubesProcessor):

    def __init__(self, cube_config: CubeConfig):
        self._cube_config = cube_config

    def process_cubes(self, cubes: Sequence[xr.Dataset]) -> xr.Dataset:
        with observe_progress('Processing cube(s)', len(cubes) + 1) as progress:

            resampled_cubes = []
            for cube in cubes:
                resampled_cube = self._resample_cube(cube)
                resampled_cubes.append(resampled_cube)
                progress.worked(1)

            if len(resampled_cubes) > 1:
                result_cube = xr.merge(resampled_cubes)
            else:
                result_cube = resampled_cubes[0]

            # Force cube to have chunks compatible with Zarr.
            result_cube = self._rechunk_cube(result_cube)

            progress.worked(1)
            return result_cube

    def _resample_cube(self, cube: xr.Dataset):
        cube_resampler = CubeResampler(self._cube_config)
        return cube_resampler.process_cube(cube)

    def _rechunk_cube(self, cube: xr.Dataset):
        cube_rechunker = CubeRechunker(self._cube_config.chunks or {})
        return cube_rechunker.process_cube(cube)
