# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import xarray as xr

from xcube.core.gridmapping import GridMapping

from ..config import CubeConfig
from .transformer import CubeTransformer, TransformedCube


class CubeResamplerT(CubeTransformer):
    def transform_cube(
        self, cube: xr.Dataset, gm: GridMapping, cube_config: CubeConfig
    ) -> TransformedCube:
        # TODO (forman): implement me
        return cube, gm, cube_config
