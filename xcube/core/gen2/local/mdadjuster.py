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

from typing import Dict, Any

import pandas as pd
import pyproj
import xarray as xr

from xcube.core.gridmapping import CRS_CRS84
from xcube.core.gridmapping import GridMapping
from xcube.version import version
from .transformer import CubeTransformer
from .transformer import TransformedCube
from ..config import CubeConfig


class CubeMetadataAdjuster(CubeTransformer):
    """Adjust a cube's metadata."""

    def transform_cube(self,
                       cube: xr.Dataset,
                       gm: GridMapping,
                       cube_config: CubeConfig) -> TransformedCube:
        history = cube.attrs.get('history')
        if isinstance(history, str):
            history = [history]
        elif isinstance(history, (list, tuple)):
            history = list(history)
        else:
            history = []
        history.append(
            dict(
                program=f'xcube gen2, version {version}',
                cube_config=cube_config.to_dict(),
            )
        )
        cube = cube.assign_attrs(
            Conventions='CF-1.7',
            history=history,
            date_created=pd.Timestamp.now().isoformat(),
            # TODO: adjust temporal metadata too
            **get_geospatial_attrs(gm)
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
    def _check_for_self_destruction(metadata: Dict[str, Any]):
        key = 'inverse_fine_structure_constant'
        value = 137
        if key in metadata and metadata[key] != value:
            # Note, this is an easter egg that causes
            # an intended internal error for testing
            raise ValueError(f'{key} must be {value}'
                             f' or running in wrong universe')


def get_geospatial_attrs(gm: GridMapping) -> Dict[str, Any]:
    if gm.crs.is_geographic:
        lon_min, lat_min, lon_max, lat_max = gm.xy_bbox
        lon_res, lat_res = gm.xy_res
    else:
        x1, y1, x2, y2 = gm.xy_bbox
        x_res, y_res = gm.xy_res
        # center position
        xm1 = (x1 + x2) / 2
        ym1 = (y1 + y2) / 2
        # center position + delta
        xm2 = xm1 + x_res
        ym2 = ym1 + y_res
        transformer = pyproj.Transformer.from_crs(crs_from=gm.crs,
                                                  crs_to=CRS_CRS84)
        xx, yy = transformer.transform((x1, x2, xm1, xm2),
                                       (y1, y2, ym1, ym2))
        lon_min, lon_max, lon_m1, lon_m2 = xx
        lat_min, lat_max, lat_m1, lat_m2 = yy
        # Estimate resolution (note, this may be VERY wrong)
        lon_res = abs(lon_m2 - lon_m1)
        lat_res = abs(lat_m2 - lat_m1)
    return dict(
        geospatial_lon_units='degrees_east',
        geospatial_lon_min=lon_min,
        geospatial_lon_max=lon_max,
        geospatial_lon_resolution=lon_res,
        geospatial_lat_units='degrees_north',
        geospatial_lat_min=lat_min,
        geospatial_lat_max=lat_max,
        geospatial_lat_resolution=lat_res,
        geospatial_bounds_crs='CRS84',
        geospatial_bounds=f'POLYGON(('
                          f'{lon_min} {lat_min}, '
                          f'{lon_min} {lat_max}, '
                          f'{lon_max} {lat_max}, '
                          f'{lon_max} {lat_min}, '
                          f'{lon_min} {lat_min}'
                          f'))',
    )
