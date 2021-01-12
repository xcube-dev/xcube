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

from typing import Optional, Union, Tuple

import pyproj
import xarray as xr

from .base import GridMapping
from .cfconv import get_dataset_grid_mappings
from .coords import from_coords


def from_dataset(dataset: xr.Dataset,
                 *,
                 tile_size: Union[int, Tuple[int, int]] = None,
                 prefer_regular: bool = True,
                 prefer_crs: pyproj.crs.CRS = None,
                 emit_warnings: bool = False) -> Optional[GridMapping]:
    grid_mappings = get_dataset_grid_mappings(dataset, emit_warnings=emit_warnings).values()
    grid_mappings = [from_coords(x_coords=grid_mapping.coords.x,
                                 y_coords=grid_mapping.coords.y,
                                 crs=grid_mapping.crs,
                                 tile_size=tile_size)
                     for grid_mapping in grid_mappings]

    # If prefer_is_rectified, try finding a rectified one
    for grid_mapping in grid_mappings:
        if prefer_regular and grid_mapping.is_regular:
            return grid_mapping

    # If prefer_crs, try finding one with that CRS
    for grid_mapping in grid_mappings:
        if prefer_crs is not None and grid_mapping.crs == prefer_crs:
            return grid_mapping

    # Get arbitrary one (here: first)
    return grid_mappings[0] if grid_mappings else None
