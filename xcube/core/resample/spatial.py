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

import dask.array as da
import numpy as np
import xarray as xr
from dask_image import ndinterp

NDImage = Union[np.ndarray, da.Array]
Aggregator = Callable[[NDImage], NDImage]


def resample_in_space(cube: xr.Dataset) -> xr.Dataset:
    return cube


def resample_ndimage(im: NDImage,
                     scale: float = 1,
                     offset: Sequence[float] = None,
                     shape: Sequence[int] = None,
                     chunks: Sequence[int] = None,
                     spline_order: int = 1,
                     aggregator: Optional[Aggregator] = np.nanmean,
                     recover_nan: bool = False) -> da.Array:
    im = _normalize_image(im)
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
        dims = len(im.shape)
        axes = {dims - 2: divisor, dims - 1: divisor}
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
    if isinstance(im, da.Array):
        return im
    return da.from_array(im)


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
