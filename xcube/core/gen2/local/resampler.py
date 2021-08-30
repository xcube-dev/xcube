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

from typing import Sequence, Tuple

import pyproj.crs
import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.core.resampling import resample_in_space
from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_instance
from xcube.util.assertions import assert_true
from .processor import DatasetTransformer
from ..config import CubeConfig


class CubeResampler(DatasetTransformer):
    @classmethod
    def new(cls,
            cubes: Sequence[Tuple[xr.Dataset, GridMapping]],
            cube_config: CubeConfig) -> 'CubeResampler':
        assert_given(cubes, 'cubes')
        assert_true(len(cubes) > 0, 'cubes must be a non-empty sequence')
        assert_instance(cube_config, CubeConfig, 'cube_config')

        identity = _cube_config_has_spatial_props(cube_config) \
                   or len(cubes) > 1
        first_cube, source_gm = cubes[0]
        return CubeResampler(first_cube, source_gm, cube_config,
                             identity=identity)

    def __init__(self,
                 first_cube: xr.Dataset,
                 source_gm: GridMapping,
                 cube_config: CubeConfig,
                 identity: bool = False):
        source_gm = source_gm.to_regular() \
            if not source_gm.is_regular else source_gm
        target_gm = source_gm \
            if identity \
            else _get_target_grid_mapping(source_gm, cube_config)
        self._first_cube = first_cube
        self._source_gm = source_gm
        self._target_gm = target_gm

    @property
    def source_gm(self) -> GridMapping:
        return self._source_gm

    @property
    def target_gm(self) -> GridMapping:
        return self._target_gm

    def transform_dataset(self,
                          cube: xr.Dataset,
                          gm: GridMapping) -> Tuple[xr.Dataset, GridMapping]:
        if self._source_gm is self._target_gm:
            return cube, self._target_gm
        gm = gm.to_regular() if not gm.is_regular else gm
        return resample_in_space(
            cube,
            source_gm=gm,
            target_gm=self._target_gm
        ), self._target_gm


def _cube_config_has_spatial_props(cube_config: CubeConfig) -> bool:
    return any(v is not None for v in (cube_config.spatial_res,
                                       cube_config.bbox,
                                       cube_config.crs))


def _get_target_grid_mapping(source_gm: GridMapping,
                             cube_config: CubeConfig) -> GridMapping:
    assert_true(source_gm.is_regular, 'source_gm must be regular')
    if cube_config.spatial_res is not None:
        xy_res = (cube_config.spatial_res, cube_config.spatial_res)
    else:
        xy_res = source_gm.xy_res
    if cube_config.bbox is not None:
        x_res, y_res = xy_res
        x_min, y_min, x_max, y_max = cube_config.bbox
        xy_min = x_min, y_min
        size = round((x_max - x_min) / x_res), round((y_max - y_min) / y_res)
    else:
        xy_min = source_gm.x_min, source_gm.y_min
        size = source_gm.size
    if cube_config.crs is not None:
        crs = pyproj.crs.CRS.from_string(cube_config.crs)
    else:
        crs = source_gm.crs
    target_gm = GridMapping.regular(size=size,
                                    xy_min=xy_min,
                                    xy_res=xy_res,
                                    crs=crs,
                                    tile_size=source_gm.tile_size,
                                    is_j_axis_up=source_gm.is_j_axis_up)
    return target_gm.derive(xy_var_names=source_gm.xy_var_names,
                            xy_dim_names=source_gm.xy_dim_names)
