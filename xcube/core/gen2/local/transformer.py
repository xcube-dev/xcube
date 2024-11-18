# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from abc import ABC, abstractmethod
from typing import Tuple

import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.util.progress import observe_progress
from .helpers import is_empty_cube
from .helpers import strip_cube
from ..config import CubeConfig

TransformedCube = tuple[xr.Dataset, GridMapping, CubeConfig]


class CubeTransformer(ABC):
    @abstractmethod
    def transform_cube(
        self, cube: xr.Dataset, gm: GridMapping, cube_config: CubeConfig
    ) -> TransformedCube:
        """
        Transform given *cube*, grid mapping *gm*, and cube
        configuration *cube_config* into a potentially new cube,
        a new grid mapping, and less restrictive cube configuration
        *cube_config*.

        The latter is achieved by returning a derived *cube_config* where
        all properties that have been "consumed" by this transformer
        are removed. See :meth:`CubeConfig.drop_property`.
        """


class CubeIdentity(CubeTransformer):
    def transform_cube(
        self, cube: xr.Dataset, gm: GridMapping, cube_config: CubeConfig
    ) -> TransformedCube:
        """Return *cube*, grid mapping *gm*, and parameters without change."""
        return cube, gm, cube_config


def transform_cube(
    t_cube: TransformedCube, transformer: CubeTransformer, label: str = ""
) -> TransformedCube:
    empty_cube = is_empty_cube(t_cube[0])
    identity = isinstance(transformer, CubeIdentity)
    if not label:
        label = f"{type(transformer).__name__}"
    if identity:
        label += " (step not applicable)"
    elif empty_cube:
        label += " (step not applicable, empty cube)"

    with observe_progress(label, 1) as progress:
        if not (identity or empty_cube):
            t_cube = transformer.transform_cube(*t_cube)
            t_cube = strip_cube(t_cube[0]), t_cube[1], t_cube[2]
        progress.worked(1)

    return t_cube
