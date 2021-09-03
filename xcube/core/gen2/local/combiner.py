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

from xcube.util.progress import observe_progress
from .transformer import TransformedCube
from ..config import CubeConfig


class CubesCombiner:
    def __init__(self, cube_config: CubeConfig):
        self._cube_config = cube_config

    # noinspection PyMethodMayBeStatic
    def combine_cubes(self, t_cubes: Sequence[TransformedCube]) \
            -> TransformedCube:
        cube, gm, _ = t_cubes[0]
        if len(t_cubes) == 1:
            return cube, gm, self._cube_config

        with observe_progress('merging cubes', 1) as observer:
            cube = xr.merge([t_cube[0] for t_cube in t_cubes])
            observer.worked(1)

        return cube, gm, self._cube_config
