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

import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.core.schema import rechunk_cube
from .transformer import CubeTransformer
from .transformer import TransformedCube
from ..config import CubeConfig


class CubeRechunker(CubeTransformer):
    """Force cube to have chunks compatible with Zarr."""

    def transform_cube(self,
                       cube: xr.Dataset,
                       gm: GridMapping,
                       cube_config: CubeConfig) -> TransformedCube:
        cube, gm = rechunk_cube(cube, gm,
                                chunks=cube_config.chunks,
                                tile_size=cube_config.tile_size)
        return cube, gm, cube_config
