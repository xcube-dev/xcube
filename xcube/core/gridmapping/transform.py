# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Union, Tuple

import numpy as np
import pyproj
import pyproj.transformer as pt
import xarray as xr

from .base import DEFAULT_TOLERANCE
from .base import GridMapping
from .coords import new_grid_mapping_from_coords
from .helpers import _assert_valid_xy_names
from .helpers import _normalize_crs


# Cannot be used, but should, see TODO in transform_grid_mapping()
#
# class TransformedGridMapping(GridMapping, abc.ABC):
#     """
#     Grid mapping constructed from 1D/2D coordinate variables and a CRS.
#     """
#
#     def __init__(self,
#                  /,
#                  xy_coords: xr.DataArray,
#                  **kwargs):
#         self._xy_coords = xy_coords
#         super().__init__(**kwargs)
#
#     @property
#     def xy_coords(self) -> xr.DataArray:
#         return self._xy_coords
#


def transform_grid_mapping(
    grid_mapping: GridMapping,
    crs: Union[str, pyproj.crs.CRS],
    *,
    tile_size: Union[int, tuple[int, int]] = None,
    xy_var_names: tuple[str, str] = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> GridMapping:
    target_crs = _normalize_crs(crs)

    if xy_var_names:
        _assert_valid_xy_names(xy_var_names, name="xy_var_names")

    source_crs = grid_mapping.crs
    if source_crs == target_crs:
        if tile_size is not None or xy_var_names is not None:
            return grid_mapping.derive(tile_size=tile_size, xy_var_names=xy_var_names)
        return grid_mapping

    transformer = pt.Transformer.from_crs(source_crs, target_crs, always_xy=True)

    def _transform(block: np.ndarray) -> np.ndarray:
        x1, y1 = block
        x2, y2 = transformer.transform(x1, y1)
        return np.stack([x2, y2])

    xy_coords = xr.apply_ufunc(
        _transform,
        grid_mapping.xy_coords,
        output_dtypes=[np.float64],
        dask="parallelized",
    )

    xy_var_names = xy_var_names or ("transformed_x", "transformed_y")

    # TODO: Use a specialized grid mapping here that can store the
    #   *xy_coords* directly. Splitting the xy_coords dask array into
    #   x,y components as done here may be very inefficient for larger
    #   arrays, because x cannot be computed independently from y.
    #   This means, any access of x chunks will cause y chunks to be
    #   computed too and vice versa. As same operations are performed
    #   on x and y arrays, this will take twice as long as if operation
    #   would be performed on the xy_coords dask array directly.

    return new_grid_mapping_from_coords(
        x_coords=xr.DataArray(xy_coords[0], name=xy_var_names[0]),
        y_coords=xr.DataArray(xy_coords[1], name=xy_var_names[1]),
        crs=target_crs,
        tile_size=tile_size,
        tolerance=tolerance,
    )
