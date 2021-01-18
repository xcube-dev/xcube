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

import math
from typing import Union, Callable, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from dask import array as da
from dask_image import ndinterp

from xcube.core.gridmapping import GridMapping
from xcube.core.gridmapping import assert_regular_grid_mapping
from xcube.core.rectify import rectify_dataset

NDImage = Union[np.ndarray, da.Array]
Aggregator = Callable[[NDImage], NDImage]


def resample_in_space(dataset: xr.Dataset,
                      source_gm: GridMapping = None,
                      target_gm: GridMapping = None):
    if source_gm is None:
        # No source grid mapping given, so do derive it from dataset
        source_gm = GridMapping.from_dataset(dataset)

    if target_gm is None:
        # No target grid mapping given, so do derive it from source
        target_gm = source_gm.to_regular()

    # target_gm must be regular
    assert_regular_grid_mapping(target_gm, name='target_gm')

    # Are source and target both geographic grid mappings?
    both_geographic = source_gm.crs.is_geographic and target_gm.crs.is_geographic

    if not both_geographic and source_gm.crs != target_gm.crs:
        # If CRSes are not both geographic and their CRSes are different
        # transform the source_gm so its CRS matches the target CRS:
        transformed_geo_coding = source_gm.to_transformed(target_gm.crs)
        return resample_in_space(dataset,
                                 source_gm=transformed_geo_coding,
                                 target_gm=target_gm)
    else:
        # If CRSes are both geographic or their CRSes are equal:
        if source_gm.is_regular:
            # If also the source is regular, then resampling reduces
            # to an affine transformation.
            return affine_transform_dataset(dataset,
                                            source_gm=source_gm,
                                            target_cm=target_gm)
        else:
            # If the source is not regular, we need to rectify it,
            # so the target is regular. Our rectification implementations
            # works only if source pixel size > target pixel size.
            # Therefore check if we must downscale source first.
            x_scale = source_gm.x_res / target_gm.x_res
            y_scale = source_gm.y_res / target_gm.y_res
            xy_scale = 0.5 * (x_scale + y_scale)
            if xy_scale > 1.25:
                # Source has lower resolution than target.
                return rectify_dataset(dataset,
                                       geo_coding=source_gm,
                                       output_geom=target_gm)
            else:
                # Source has higher resolution than target.
                # Downscale first, then rectify
                downscaled_dataset = affine_transform_dataset(dataset,
                                                              source_gm=source_gm,
                                                              target_cm=target_gm)
                x_name, y_name = source_gm.xy_var_names
                downscaled_gm = GridMapping.from_coords(downscaled_dataset[x_name],
                                                        downscaled_dataset[y_name],
                                                        source_gm.crs)
                return rectify_dataset(downscaled_dataset,
                                       geo_coding=downscaled_gm,
                                       output_geom=target_gm)


def affine_transform_dataset(dataset: xr.Dataset,
                             source_gm: GridMapping = None,
                             target_cm: GridMapping = None):
    if source_gm.crs != target_cm.crs:
        raise ValueError(f'CRS of source_gm and target_cm must be equal, '
                         f'was "{source_gm.crs.name}" and "{target_cm.crs.name}"')
    assert_regular_grid_mapping(source_gm)
    assert_regular_grid_mapping(target_cm)
    at = source_gm.ij_transform_to(target_cm)
    print('affine transform:', at)
    ((x_scale, _, x_off), (_, y_scale, y_off)) = at
    # TODO: scale may be wrong - scales can be negative if one of
    #   source_gm.is_j_axis_up or source_gm.is_j_axis_up is True
    scale = 0.5 * (abs(x_scale) + abs(y_scale))
    x_dim, y_dim = source_gm.xy_dim_names
    width, height = target_cm.size
    tile_width, tile_height = target_cm.tile_size
    yx_dims = (y_dim, x_dim)
    coords = dict()
    data_vars = dict()
    for k, var in dataset.variables.items():
        new_var = None
        if var.ndim >= 2 and var.dims[-2] == yx_dims:
            var_data = resample_ndimage(var.data,
                                        scale=scale,
                                        offset=(x_off, y_off),
                                        shape=(height, width),
                                        chunks=(tile_height, tile_width))
            new_var = xr.DataArray(var_data, dims=var.dims, attrs=var.attrs)
        elif x_dim not in var.dims and y_dim not in var.dims:
            new_var = var.copy()
        if new_var is not None:
            if k in dataset.coords:
                coords[k] = new_var
            elif k in dataset.data_vars:
                data_vars[k] = new_var
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=dataset.attrs)


def resample_ndimage(im: NDImage,
                     scale: Union[float, Tuple[float, float]] = 1,
                     offset: Sequence[float] = None,
                     shape: Sequence[int] = None,
                     chunks: Sequence[int] = None,
                     spline_order: int = 1,
                     aggregator: Optional[Aggregator] = np.nanmean,
                     recover_nan: bool = False) -> da.Array:
    im = da.asarray(im)
    offset = _normalize_offset(offset)
    if shape is None:
        shape = _resize_shape(im.shape, scale, 1)
    else:
        shape = _normalize_shape(shape, im)
    chunks = _normalize_chunks(chunks, shape)
    inv_scale = 1 / scale
    divisor = math.ceil(inv_scale)
    if divisor >= 2 and aggregator is not None:
        # Downsampling
        # ------------
        num_dims = len(im.shape)
        axes = {num_dims - 2: divisor, num_dims - 1: divisor}
        elongation = divisor / inv_scale
        larger_shape = _resize_shape(shape, divisor, divisor)
        divisible_chunks = _make_divisible_tiles(larger_shape, divisor)
        im = _transform_array(im, elongation, offset, larger_shape, divisible_chunks, spline_order, recover_nan)
        # print('Downsampling:', scale, inv_scale, divisor, elongation, \
        #                        1 / elongation, larger_shape, im.shape, im.chunks)
        im = da.coarsen(aggregator, im, axes)
        if shape != im.shape:
            im = im[..., 0:shape[-2], 0:shape[-1]]
        if chunks is not None:
            im = im.rechunk(chunks)
    else:
        # Upsampling
        # ----------
        # print('Upsampling:', scale, offset, shape, nearest)
        im = _transform_array(im, scale, offset, shape, chunks, spline_order, recover_nan)
    return im


def _transform_array(im: da.Array,
                     scale: float,
                     offset: np.ndarray,
                     shape: Tuple[int, ...],
                     chunks: Optional[Tuple[int, ...]],
                     spline_order: int,
                     recover_nan: bool) -> da.Array:
    if _is_no_op(im, scale, offset, shape):
        return im
    matrix = [
        [1 / scale, 0.0],
        [0.0, 1 / scale],
    ]
    at_kwargs = dict(
        offset=offset,
        order=spline_order,
        output_shape=shape,
        output_chunks=chunks,
        mode='constant',
    )
    if recover_nan and spline_order > 0:
        # We can "recover" values that are neighbours to NaN values
        # that would otherwise become NaN too.
        mask = da.isnan(im)
        # First check if there are NaN values ar all
        if da.any(mask):
            # Yes, then
            # 1. replace NaN by zero
            filled_im = da.where(mask, 0.0, im)
            # 2. transform the zeo-filled image
            scaled_im = ndinterp.affine_transform(filled_im, matrix, **at_kwargs, cval=0.0)
            # 3. transform the inverted mask
            scaled_norm = ndinterp.affine_transform(1.0 - mask, matrix, **at_kwargs, cval=0.0)
            # 4. put back NaN where there was zero, otherwise decode using scaled mask
            return da.where(da.isclose(scaled_norm, 0.0), np.nan, scaled_im / scaled_norm)

    # No dealing with NaN required
    return ndinterp.affine_transform(im, matrix, **at_kwargs, cval=np.nan)


def _resize_shape(shape: Sequence[int],
                  scale: float,
                  divisor: int = 1) -> Tuple[int, ...]:
    wf = shape[-1] * scale
    hf = shape[-2] * scale
    if divisor > 1:
        w = divisor * ((round(wf) + divisor - 1) // divisor)
        h = divisor * ((round(hf) + divisor - 1) // divisor)
    else:
        w = math.ceil(wf)
        h = math.ceil(hf)
    return tuple(shape[0:-2]) + (h, w)


def _make_divisible_tiles(larger_shape: Tuple[int, ...], divisor: int) -> Tuple[int, ...]:
    tile_size = divisor * ((2048 + divisor - 1) // divisor)
    w = min(larger_shape[-1], tile_size)
    h = min(larger_shape[-2], tile_size)
    return (len(larger_shape) - 2) * (1,) + (h, w)


def _normalize_image(im: NDImage) -> da.Array:
    return da.asarray(im)


def _normalize_offset(offset: Optional[Sequence[float]]) -> np.ndarray:
    if offset is None:
        offset = [0.0, 0.0]
    elif len(offset) != 2:
        raise ValueError('illegal image offset')
    return np.array(offset, dtype=np.float64)


def _normalize_shape(shape: Optional[Sequence[int]], im: NDImage) -> Tuple[int, ...]:
    if shape is None:
        return im.shape
    if len(shape) < 2 or len(shape) > len(im.shape):
        raise ValueError('illegal image shape')
    return im.shape[0:-len(shape)] + tuple(shape)


def _normalize_chunks(chunks: Optional[Sequence[int]],
                      shape: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    if chunks is None:
        return None
    if len(chunks) < 2 or len(chunks) > len(shape):
        raise ValueError('illegal image chunks')
    return (len(shape) - len(chunks)) * (1,) + tuple(chunks)


def _is_no_op(im: NDImage,
              scale: float,
              offset: np.ndarray,
              shape: Tuple[int, ...]):
    if math.isclose(scale, 1) and shape == im.shape:
        offset_y, offset_x = offset
        return math.isclose(offset_x, 0) and math.isclose(offset_y, 0)
    return False
