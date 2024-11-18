# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import pyproj.crs
import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.core.resampling import resample_in_space
from .transformer import CubeTransformer
from .transformer import TransformedCube
from ..config import CubeConfig


class CubeResamplerXY(CubeTransformer):
    def transform_cube(
        self, cube: xr.Dataset, source_gm: GridMapping, cube_config: CubeConfig
    ) -> TransformedCube:
        target_gm = _compute_target_grid_mapping(cube_config, source_gm)

        if source_gm.is_close(target_gm):
            resampled_cube = cube
        else:
            resampled_cube = resample_in_space(
                cube, source_gm=source_gm, target_gm=target_gm
            )

        cube_config = cube_config.drop_props(["crs", "spatial_res", "bbox"])

        return resampled_cube, target_gm, cube_config


def _compute_target_grid_mapping(
    cube_config: CubeConfig, source_gm: GridMapping
) -> GridMapping:
    # assert_true(source_gm.is_regular, 'source_gm must be regular')

    target_crs = cube_config.crs
    target_bbox = cube_config.bbox
    target_spatial_res = cube_config.spatial_res

    if target_crs is None and target_bbox is None and target_spatial_res is None:
        # Nothing to do
        if source_gm.is_regular:
            return source_gm
        return source_gm.to_regular(tile_size=cube_config.tile_size)

    if target_spatial_res is not None:
        xy_res = (target_spatial_res, target_spatial_res)
    else:
        xy_res = source_gm.xy_res
    if target_bbox is not None:
        x_res, y_res = xy_res
        x_min, y_min, x_max, y_max = target_bbox
        xy_min = x_min, y_min
        size = round((x_max - x_min) / x_res), round((y_max - y_min) / y_res)
    else:
        xy_min = source_gm.x_min, source_gm.y_min
        size = source_gm.size
    if target_crs is not None:
        crs = pyproj.crs.CRS.from_string(target_crs)
    else:
        crs = source_gm.crs
    target_gm = GridMapping.regular(
        size=size,
        xy_min=xy_min,
        xy_res=xy_res,
        crs=crs,
        tile_size=source_gm.tile_size,
        is_j_axis_up=source_gm.is_j_axis_up,
    )
    return target_gm.derive(
        xy_var_names=source_gm.xy_var_names, xy_dim_names=source_gm.xy_dim_names
    )
