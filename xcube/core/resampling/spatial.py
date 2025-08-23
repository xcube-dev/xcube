# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from collections.abc import Hashable, Mapping
from typing import Any, Callable, Optional, Union

import numpy as np
import xarray as xr
from dask import array as da

from xcube.core.gridmapping import GridMapping
from xcube.core.gridmapping.coords import Coords2DGridMapping
from xcube.core.gridmapping.helpers import scale_xy_res_and_size

from .affine import affine_transform_dataset, resample_dataset
from .rectify import rectify_dataset

NDImage = Union[np.ndarray, da.Array]
Aggregator = Callable[[NDImage], NDImage]

# If _SCALE_LIMIT is exceeded, we don't need
# to downscale source image before we can
# rectify it.
_SCALE_LIMIT = 0.95


def resample_in_space(
    source_ds: xr.Dataset,
    /,
    source_gm: Optional[GridMapping] = None,
    target_gm: Optional[GridMapping] = None,
    ref_ds: Optional[xr.Dataset] = None,
    spline_orders: Optional[int | Mapping[type | str, int]] = None,
    agg_methods: Optional[str | Mapping[type | str, str]] = None,
    recover_nan: Optional[bool | Mapping[type | str, bool]] = False,
):
    """
    Resample a dataset *source_ds* in the spatial dimensions.

    Args:
        source_ds: The source dataset. Data variables must have
            dimensions in the following order: optional 3rd dimension followed
            by the y-dimension (e.g., `y` or `lat`) followed by the
            x-dimension (e.g., `x` or `lon`).
        source_gm: The source grid mapping.
        target_gm: The target grid mapping. Must be regular.
        ref_ds: An optional dataset that provides the
            target grid mapping if *target_gm* is not provided.
            If *ref_ds* is given, its coordinate variables are copied
            by reference into the returned dataset.
        spline_orders: Spline orders to be used for upsampling
            spatial data variables. It can be a single spline order
            for all variables or a dictionary that maps a variable name or a data dtype
            to the spline order. A spline order is given by one of `0`
            (nearest neighbor), `1` (linear), `2` (bi-linear), or `3` (cubic).
            The default is `3` fo floating point datasets and `0` for integer datasets.
        agg_methods: Aggregation methods to be used for downsampling
            spatial data variables. It can be a single aggregation method for all
            variables or a dictionary that maps a variable name or a data dtype to the
            aggregation method. The aggregation method is a function like `np.sum`,
            `np.mean` which is propagated to [`dask.array.coarsen`](https://docs.dask.org/en/stable/generated/dask.array.coarsen.html).
        recover_nan: If true, whether a special algorithm shall be used that is able
            to recover values that would otherwise yield NaN during resampling. Default
            is False for all variable types since this may require considerable CPU
            resources on top. It can be a single aggregation method for all
            variables or a dictionary that maps a variable name or a data dtype to a
            boolean.

    Returns:
        The spatially resampled dataset, or None if the requested output area does
        not intersect with *dataset*.

    Notes:
        - If the source grid mapping *source_gm* is not given, it is derived from *dataset*:
          `source_gm = GridMapping.from_dataset(source_ds)`.
        - If the target grid mapping *target_gm* is not given, it is derived from
          *ref_ds* as `target_gm = GridMapping.from_dataset(ref_ds)`; if *ref_ds* is
          not given, *target_gm* is derived from *source_gm* as
          `target_gm = source_gm.to_regular()`.
        - New in 1.6: If *ref_ds* is given, its coordinate variables are copied by
          reference into the returned dataset.
        - If *source_gm* is almost equal to *target_gm*, this function is a no-op
          and *dataset* is returned unchanged.
        - further information is given in the [xcube documentation](https://xcube.readthedocs.io/en/latest/rectify.html)
    """
    if source_gm is None:
        # No source grid mapping given, so do derive it from dataset.
        source_gm = GridMapping.from_dataset(source_ds)

    if target_gm is None:
        # No target grid mapping given, so do derive it
        # from target dataset or source grid mapping.
        if ref_ds is not None:
            target_gm = GridMapping.from_dataset(ref_ds)
        else:
            target_gm = source_gm.to_regular()

    if source_gm.is_close(target_gm):
        # If source and target grid mappings are almost equal.
        return source_ds

    # target_gm must be regular
    GridMapping.assert_regular(target_gm, name="target_gm")

    # Are source and target both geographic grid mappings?
    both_geographic = source_gm.crs.is_geographic and target_gm.crs.is_geographic

    if both_geographic or source_gm.crs == target_gm.crs:
        # If CRSes are both geographic or their CRSes are equal:
        if source_gm.is_regular:
            # If also the source is regular, then resampling reduces
            # to an affine transformation.
            return affine_transform_dataset(
                source_ds,
                source_gm=source_gm,
                ref_ds=ref_ds,
                target_gm=target_gm,
                var_configs=var_configs,
                encode_cf=encode_cf,
                gm_name=gm_name,
            )

        # If the source is not regular, we need to rectify it,
        # so the target is regular. Our rectification implementation
        # works only correctly if source pixel size >= target pixel
        # size. Therefore, check if we must downscale source first.
        x_scale = source_gm.x_res / target_gm.x_res
        y_scale = source_gm.y_res / target_gm.y_res
        if x_scale > _SCALE_LIMIT and y_scale > _SCALE_LIMIT:
            # Source pixel size >= target pixel size.
            # We can rectify.
            return rectify_dataset(
                source_ds,
                source_gm=source_gm,
                ref_ds=ref_ds,
                target_gm=target_gm,
                encode_cf=encode_cf,
                gm_name=gm_name,
                **(rectify_kwargs or {}),
            )
        else:
            # Source has higher resolution than target.
            # Downscale first, then rectify
            _, downscaled_size = scale_xy_res_and_size(
                source_gm.xy_res, source_gm.size, (x_scale, y_scale)
            )
            downscaled_dataset = resample_dataset(
                source_ds,
                ((1 / x_scale, 0, 0), (0, 1 / y_scale, 0)),
                size=downscaled_size,
                tile_size=source_gm.tile_size,
                xy_dim_names=source_gm.xy_dim_names,
                var_configs=var_configs,
            )
            downscaled_gm = GridMapping.from_dataset(
                downscaled_dataset,
                tile_size=source_gm.tile_size,
                crs=source_gm.crs,
            )
            return rectify_dataset(
                downscaled_dataset,
                source_gm=downscaled_gm,
                ref_ds=ref_ds,
                target_gm=target_gm,
                encode_cf=encode_cf,
                gm_name=gm_name,
                **(rectify_kwargs or {}),
            )

    # If CRSes are not both geographic and their CRSes are different
    # transform the source_gm so its CRS matches the target CRS:
    transformed_source_gm = source_gm.transform(target_gm.crs, xy_res=target_gm.xy_res)
    if not isinstance(source_gm, Coords2DGridMapping):
        source_ds = source_ds.drop_vars(source_gm.xy_dim_names)
    list_grid_mapping = []
    for var in source_ds.data_vars:
        if "grid_mapping" in source_ds[var].attrs:
            attrs = source_ds[var].attrs
            list_grid_mapping.append(attrs["grid_mapping"])
            del attrs["grid_mapping"]
            source_ds[var] = source_ds[var].assign_attrs(attrs)
    source_ds = source_ds.drop_vars(list_grid_mapping)
    if "crs" in source_ds:
        source_ds = source_ds.drop_vars("crs")
    if "spatial_ref" in source_ds:
        source_ds = source_ds.drop_vars("spatial_ref")
    source_ds = source_ds.copy()
    transformed_x, transformed_y = transformed_source_gm.xy_coords
    attrs = dict(grid_mapping="spatial_ref")
    transformed_x.attrs = attrs
    transformed_y.attrs = attrs
    source_ds = source_ds.assign_coords(
        spatial_ref=xr.DataArray(0, attrs=transformed_source_gm.crs.to_cf()),
        transformed_x=transformed_x,
        transformed_y=transformed_y,
    )
    return resample_in_space(
        source_ds,
        source_gm=transformed_source_gm,
        ref_ds=ref_ds,
        target_gm=target_gm,
        var_configs=var_configs,
        encode_cf=encode_cf,
        gm_name=gm_name,
        rectify_kwargs=rectify_kwargs,
    )
