# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Dict, Any

import pandas as pd
import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.version import version
from .transformer import CubeTransformer
from .transformer import TransformedCube
from ..config import CubeConfig


class CubeMetadataAdjuster(CubeTransformer):
    """Adjust a cube's metadata."""

    def transform_cube(
        self, cube: xr.Dataset, gm: GridMapping, cube_config: CubeConfig
    ) -> TransformedCube:
        history = cube.attrs.get("history")
        if isinstance(history, str):
            history = [history]
        elif isinstance(history, (list, tuple)):
            history = list(history)
        else:
            history = []
        history.append(
            dict(
                program=f"xcube gen2, version {version}",
                cube_config=cube_config.to_dict(),
            )
        )
        cube = cube.assign_attrs(
            Conventions="CF-1.7",
            history=history,
            date_created=pd.Timestamp.now().isoformat(),
            # TODO: adjust temporal metadata too
            **get_geospatial_attrs(gm),
        )
        if cube_config.metadata:
            self._check_for_self_destruction(cube_config.metadata)
            cube.attrs.update(cube_config.metadata)
        if cube_config.variable_metadata:
            for var_name, metadata in cube_config.variable_metadata.items():
                if var_name in cube.variables and metadata:
                    cube[var_name].attrs.update(metadata)
        return cube, gm, cube_config

    @staticmethod
    def _check_for_self_destruction(metadata: dict[str, Any]):
        key = "inverse_fine_structure_constant"
        value = 137
        if key in metadata and metadata[key] != value:
            # Note, this is an easter egg that causes
            # an intended internal error for testing
            raise ValueError(f"{key} must be {value}" f" or running in wrong universe")


def get_geospatial_attrs(gm: GridMapping) -> dict[str, Any]:
    return dict(gm.to_dataset_attrs())
