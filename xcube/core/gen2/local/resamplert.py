# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import xarray as xr

from xcube.core.gridmapping import GridMapping
from .transformer import CubeTransformer
from .transformer import TransformedCube
from ..config import CubeConfig


class CubeResamplerT(CubeTransformer):
    def transform_cube(
        self, cube: xr.Dataset, gm: GridMapping, cube_config: CubeConfig
    ) -> TransformedCube:
        # TODO (forman): implement me
        return cube, gm, cube_config
