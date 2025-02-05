# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import math
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Callable, Optional, Union

import numpy as np
import xarray as xr
from dask import array as da
from dask_image import ndinterp

from xcube.core.gridmapping import GridMapping
from xcube.core.gridmapping.helpers import AffineTransformMatrix
from xcube.util.assertions import assert_true

from .cf import complete_resampled_dataset

NDImage = Union[np.ndarray, da.Array]
Aggregator = Callable[[NDImage], NDImage]


def affine_transform_dataset(
    source_ds: xr.Dataset,
    /,
    source_gm: Optional[GridMapping] = None,
    target_gm: Optional[GridMapping] = None,
    ref_ds: Optional[xr.Dataset] = None,
    var_configs: Optional[Mapping[Hashable, Mapping[str, Any]]] = None,
    encode_cf: bool = True,
    gm_name: Optional[str] = None,
    reuse_coords: bool = False,
) -> xr.Dataset:
    """Resample dataset according to an affine transformation.

    The affine transformation will be applied only if the CRS of
    *source_gm* and the CRS of *target_gm* are both geographic or equal.
    Otherwise, a ``ValueError`` will be raised.

    New in 1.6: If *target_ds* is given, its coordinate
    variables are copied by reference into the returned
    dataset.

    Args:
        source_ds: The source dataset
        source_gm: Optional source grid mapping of *dataset*.
            If not provided, computed from *source_ds*.
            Must be regular and must have same CRS as *target_gm*.
        target_gm: Optional target grid mapping. If not provided,
            computed from *target_ds* or source grid mapping.
            Must be regular and must have same CRS as source grid mapping.
        ref_ds: An optional dataset that provides the
            target grid mapping if *target_gm* is not provided.
            If *ref_ds* is given, its coordinate variables are copied
            by reference into the returned dataset.
        var_configs: Optional resampling configurations for individual
            variables.
        encode_cf: Whether to encode the target grid mapping into the
            resampled dataset in a CF-compliant way. Defaults to
            ``True``.
        gm_name: Name for the grid mapping variable. Defaults to "crs".
            Used only if *encode_cf* is ``True``.
        reuse_coords: Whether to either reuse target coordinate arrays
            from target_gm or to compute new ones.

    Returns:
        The resampled target dataset.
    """
    if source_gm is None:
        # No source grid mapping given, so do derive it from dataset
        source_gm = GridMapping.from_dataset(source_ds)

    if target_gm is None:
        # No target grid mapping given, so do derive it
        # from reference dataset or source grid mapping
        if ref_ds is not None:
            target_gm = GridMapping.from_dataset(ref_ds)
        else:
            target_gm = source_gm.to_regular()

    # Are source and target both geographic grid mappings?
    both_geographic = source_gm.crs.is_geographic and target_gm.crs.is_geographic
    if not (both_geographic or source_gm.crs == target_gm.crs):
        raise ValueError(
            f"CRS of source_gm and target_gm must be equal,"
            f' was "{source_gm.crs.name}"'
            f' and "{target_gm.crs.name}"'
        )
    GridMapping.assert_regular(source_gm, name="source_gm")
    GridMapping.assert_regular(target_gm, name="target_gm")
    resampled_dataset = resample_dataset(
        dataset=source_ds,
        matrix=target_gm.ij_transform_to(source_gm),
        size=target_gm.size,
        tile_size=target_gm.tile_size,
        xy_dim_names=source_gm.xy_dim_names,
        var_configs=var_configs,
    )
    has_bounds = any(
        source_ds[var_name].attrs.get("bounds") for var_name in source_gm.xy_var_names
    )
    new_coords = target_gm.to_coords(
        xy_var_names=source_gm.xy_var_names,
        xy_dim_names=source_gm.xy_dim_names,
        exclude_bounds=not has_bounds,
        reuse_coords=reuse_coords,
    )
    return complete_resampled_dataset(
        encode_cf,
        resampled_dataset.assign_coords(new_coords),
        target_gm,
        gm_name,
        ref_ds.coords if ref_ds else None,
    )


def resample_dataset(
    dataset: xr.Dataset,
    matrix: AffineTransformMatrix,
    size: tuple[int, int],
    tile_size: tuple[int, int],
    xy_dim_names: tuple[str, str],
    var_configs: Mapping[Hashable, Mapping[str, Any]] = None,
) -> xr.Dataset:
    """Resample dataset according to an affine transformation.

    Args:
        dataset: The source dataset
        matrix: Affine transformation matrix.
        size: Target image size.
        tile_size: Target image tile size.
        xy_dim_names: Names of the spatial dimensions.
        var_configs: Optional resampling configurations for individual
            variables.

    Returns:
        The resampled target dataset.
    """
    ((i_scale, _, i_off), (_, j_scale, j_off)) = matrix
    width, height = size
    tile_width, tile_height = tile_size
    x_dim, y_dim = xy_dim_names
    yx_dims = (y_dim, x_dim)
    coords = dict()
    var_configs = var_configs or {}
    data_vars = dict()
    for k, var in dataset.variables.items():
        new_var = None
        if var.ndim >= 2 and var.dims[-2:] == yx_dims:
            var_config = var_configs.get(k, dict())
            if np.issubdtype(var.dtype, np.integer) or np.issubdtype(var.dtype, bool):
                spline_order = 0
                aggregator = None
                recover_nan = False
            else:
                spline_order = 1
                aggregator = np.nanmean
                # forman: changed default from True to False (v1.6, 2024-06-05)
                recover_nan = False
            var_data = resample_ndimage(
                var.data,
                scale=(j_scale, i_scale),
                offset=(j_off, i_off),
                shape=(height, width),
                chunks=(tile_height, tile_width),
                spline_order=var_config.get("spline_order", spline_order),
                aggregator=var_config.get("aggregator", aggregator),
                recover_nan=var_config.get("recover_nan", recover_nan),
            )
            new_var = xr.DataArray(var_data, dims=var.dims, attrs=var.attrs)
        elif x_dim not in var.dims and y_dim not in var.dims:
            new_var = var.copy()
        if new_var is not None:
            if k in dataset.coords:
                coords[k] = new_var
            elif k in dataset.data_vars:
                data_vars[k] = new_var

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=dataset.attrs)


def resample_ndimage(
    image: NDImage,
    scale: Union[float, tuple[float, float]] = 1,
    offset: Union[float, tuple[float, float]] = None,
    shape: Union[int, tuple[int, int]] = None,
    chunks: Sequence[int] = None,
    spline_order: int = 1,
    aggregator: Optional[Aggregator] = np.nanmean,
    recover_nan: bool = False,
) -> da.Array:
    image = da.asarray(image)
    offset = _normalize_offset(offset, image.ndim)
    scale = _normalize_scale(scale, image.ndim)
    if shape is None:
        shape = resize_shape(image.shape, scale)
    else:
        shape = _normalize_shape(shape, image)
    chunks = _normalize_chunks(chunks, shape)
    scale_y, scale_x = scale[-2], scale[-1]
    divisor_x = math.ceil(abs(scale_x))
    divisor_y = math.ceil(abs(scale_y))
    if (divisor_x >= 2 or divisor_y >= 2) and aggregator is not None:
        # Downsampling
        # ------------
        axes = {image.ndim - 2: divisor_y, image.ndim - 1: divisor_x}
        elongation = _normalize_scale(
            (scale_y / divisor_y, scale_x / divisor_x), image.ndim
        )
        larger_shape = resize_shape(
            shape, (divisor_y, divisor_x), divisor_x=divisor_x, divisor_y=divisor_y
        )
        # print('Downsampling: ', scale)
        # print('  divisor:', (divisor_y, divisor_x))
        # print('  elongation:', elongation)
        # print('  shape:', shape)
        # print('  larger_shape:', larger_shape)
        divisible_chunks = _make_divisible_tiles(larger_shape, divisor_x, divisor_y)
        image = _transform_array(
            image,
            elongation,
            offset,
            larger_shape,
            divisible_chunks,
            spline_order,
            recover_nan,
        )
        image = da.coarsen(aggregator, image, axes)
        if shape != image.shape:
            image = image[..., 0 : shape[-2], 0 : shape[-1]]
        if chunks is not None:
            image = image.rechunk(chunks)
    else:
        # Upsampling
        # ----------
        # print('Upsampling: ', scale)
        image = _transform_array(
            image, scale, offset, shape, chunks, spline_order, recover_nan
        )
    return image


def _transform_array(
    image: da.Array,
    scale: tuple[float, ...],
    offset: tuple[float, ...],
    shape: tuple[int, ...],
    chunks: Optional[tuple[int, ...]],
    spline_order: int,
    recover_nan: bool,
) -> da.Array:
    """Apply affine transformation to ND-image.

    Args:
        image: ND-image with shape (..., size_y, size_x)
        scale: Scaling factors (1, ..., 1, sy, sx)
        offset: Offset values (0, ..., 0, oy, ox)
        shape: (..., size_y, size_x)
        chunks: (..., chunk_size_y, chunk_size_x)
        spline_order: 0 ... 5
        recover_nan: True/False

    Returns:
        Transformed ND-image.
    """
    assert_true(len(scale) == image.ndim, "invalid scale")
    assert_true(len(offset) == image.ndim, "invalid offset")
    assert_true(len(shape) == image.ndim, "invalid shape")
    assert_true(chunks is None or len(chunks) == image.ndim, "invalid chunks")
    if _is_no_op(image, scale, offset, shape):
        return image
    # As of scipy 0.18, matrix = scale is no longer supported.
    # Therefore we use the diagonal matrix form here,
    # where scale is the diagonal.
    matrix = np.diag(scale)
    at_kwargs = dict(
        offset=offset,
        order=spline_order,
        output_shape=shape,
        output_chunks=chunks,
        mode="constant",
    )
    if recover_nan and spline_order > 0:
        # We can "recover" values that are neighbours to NaN values
        # that would otherwise become NaN too.
        mask = da.isnan(image)
        # First check if there are NaN values ar all
        if da.any(mask):
            # Yes, then
            # 1. replace NaN by zero
            filled_im = da.where(mask, 0.0, image)
            # 2. transform the zero-filled image
            scaled_im = ndinterp.affine_transform(
                filled_im, matrix, **at_kwargs, cval=0.0
            )
            # 3. transform the inverted mask
            scaled_norm = ndinterp.affine_transform(
                1.0 - mask, matrix, **at_kwargs, cval=0.0
            )
            # 4. put back NaN where there was zero,
            #    otherwise decode using scaled mask
            return da.where(
                da.isclose(scaled_norm, 0.0), np.nan, scaled_im / scaled_norm
            )

    # No dealing with NaN required
    return ndinterp.affine_transform(image, matrix, **at_kwargs, cval=np.nan)


def resize_shape(
    shape: Sequence[int],
    scale: Union[float, tuple[float, ...]],
    divisor_x: int = 1,
    divisor_y: int = 1,
) -> tuple[int, ...]:
    scale = _normalize_scale(scale, len(shape))
    height, width = shape[-2], shape[-1]
    scale_y, scale_x = scale[-2], scale[-1]
    wf = width * abs(scale_x)
    hf = height * abs(scale_y)
    w = divisor_x * math.ceil(wf / divisor_x)
    h = divisor_y * math.ceil(hf / divisor_y)
    return tuple(shape[0:-2]) + (h, w)


def _make_divisible_tiles(
    larger_shape: tuple[int, ...], divisor_x: int, divisor_y: int
) -> tuple[int, ...]:
    w = min(larger_shape[-1], divisor_x * ((2048 + divisor_x - 1) // divisor_x))
    h = min(larger_shape[-2], divisor_y * ((2048 + divisor_y - 1) // divisor_y))
    return (len(larger_shape) - 2) * (1,) + (h, w)


def _normalize_image(im: NDImage) -> da.Array:
    return da.asarray(im)


def _normalize_offset(
    offset: Optional[Sequence[float]], ndim: int
) -> tuple[float, ...]:
    return _normalize_pair(offset, 0.0, ndim, "offset")


def _normalize_scale(scale: Optional[Sequence[float]], ndim: int) -> tuple[float, ...]:
    return _normalize_pair(scale, 1.0, ndim, "scale")


def _normalize_pair(
    pair: Optional[Sequence[float]], default: float, ndim: int, name: str
) -> tuple[float, ...]:
    if pair is None:
        pair = [default, default]
    elif isinstance(pair, (int, float)):
        pair = [pair, pair]
    elif len(pair) != 2:
        raise ValueError(f"illegal image {name}")
    return (ndim - 2) * (default,) + tuple(pair)


def _normalize_shape(shape: Optional[Sequence[int]], im: NDImage) -> tuple[int, ...]:
    if shape is None:
        return im.shape
    if len(shape) != 2:
        raise ValueError("illegal image shape")
    return im.shape[0:-2] + tuple(shape)


def _normalize_chunks(
    chunks: Optional[Sequence[int]], shape: tuple[int, ...]
) -> Optional[tuple[int, ...]]:
    if chunks is None:
        return None
    if len(chunks) < 2 or len(chunks) > len(shape):
        raise ValueError("illegal image chunks")
    return (len(shape) - len(chunks)) * (1,) + tuple(chunks)


def _is_no_op(
    im: NDImage, scale: Sequence[float], offset: Sequence[float], shape: tuple[int, ...]
):
    return (
        shape == im.shape
        and all(math.isclose(s, 1) for s in scale)
        and all(math.isclose(o, 0) for o in offset)
    )
