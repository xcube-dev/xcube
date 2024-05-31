# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from collections.abc import Sequence

import xarray as xr

from xcube.util.progress import observe_progress
from .transformer import TransformedCube
from ..config import CubeConfig


class CubesCombiner:
    def __init__(self, cube_config: CubeConfig):
        self._cube_config = cube_config

    # noinspection PyMethodMayBeStatic
    def combine_cubes(self, t_cubes: Sequence[TransformedCube]) -> TransformedCube:
        cube, gm, _ = t_cubes[0]
        if len(t_cubes) == 1:
            return cube, gm, self._cube_config

        with observe_progress("merging cubes", 1) as observer:
            cube = xr.merge([t_cube[0] for t_cube in t_cubes])
            observer.worked(1)

        return cube, gm, self._cube_config
