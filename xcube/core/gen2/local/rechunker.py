# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.core.schema import rechunk_cube

from ..config import CubeConfig
from .transformer import CubeTransformer, TransformedCube


class CubeRechunker(CubeTransformer):
    """Force cube to have chunks compatible with Zarr."""

    def transform_cube(
        self, cube: xr.Dataset, gm: GridMapping, cube_config: CubeConfig
    ) -> TransformedCube:
        cube, gm = rechunk_cube(
            cube, gm, chunks=cube_config.chunks, tile_size=cube_config.tile_size
        )
        return cube, gm, cube_config
