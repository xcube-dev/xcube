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

import dask.array
import numpy as np
import xarray as xr
from dask_image import ndinterp

NDImage = Union[np.ndarray, dask.array.Array]
Aggregator = Callable[[NDImage], NDImage]


def resample_in_space(cube: xr.Dataset) -> xr.Dataset:
    return cube


def resample_array(im: NDImage,
                   scale: float = 1,
                   offset: Sequence[int] = None,
                   shape: Sequence[int] = None,
                   nearest: bool = False,
                   aggregator: Optional[Aggregator] = np.nanmean) -> dask.array.Array:
    im = _normalize_image(im)
    offset = _normalize_offset(offset)
    if shape is None:
        shape = _resize_shape(im.shape, scale, divisor=1)
    else:
        shape = _normalize_shape(shape, im)
    inv_scale = 1 / scale
    divisor = math.ceil(inv_scale)
    if divisor >= 2 and aggregator is not None:
        dims = len(im.shape)
        axes = {dims - 2: divisor, dims - 1: divisor}
        elongation = divisor / inv_scale
        larger_shape = _resize_shape(shape, scale=divisor, divisor=divisor)
        im = _transform_array(im, elongation, offset, shape=larger_shape, nearest=nearest)
        print('Downsampling:', scale, inv_scale, divisor, elongation, 1 / elongation, larger_shape, im.shape, im.chunks)
        im = dask.array.coarsen(aggregator, im, axes)
        if shape != im.shape:
            im = im[..., 0:shape[-2], 0:shape[-1]]
        return im
    else:
        print('Upsampling:', scale, offset, shape, nearest)
        return _transform_array(im, scale, offset, shape, nearest)


def _transform_array(im: dask.array.Array,
                     scale: float,
                     offset: np.ndarray,
                     shape: Tuple[int, ...],
                     nearest: bool) -> dask.array.Array:
    if _is_no_op(im, scale, offset, shape):
        return im
    inv_scale = 1 / scale
    matrix = [
        [inv_scale, 0.0],
        [0.0, inv_scale],
    ]
    at_kwargs = dict(
        offset=np.array([offset[0], offset[1]]) if offset is not None else np.array([0.0, 0.0]),
        order=0 if nearest else 1,
        output_shape=shape,
        output_chunks=shape,
        mode='constant',
    )
    mask = dask.array.isnan(im)
    if dask.array.any(mask):
        filled_im = dask.array.where(mask, 0.0, im)
        scaled_im = ndinterp.affine_transform(filled_im, matrix, **at_kwargs, cval=0.0)
        scaled_norm = ndinterp.affine_transform(1.0 - mask, matrix, **at_kwargs, cval=0.0)
        return dask.array.where(dask.array.isclose(scaled_norm, 0.0), np.nan, scaled_im / scaled_norm)
    else:
        return ndinterp.affine_transform(im, matrix, **at_kwargs, cval=np.nan)


def _resize_shape(shape: Sequence[int],
                  scale: float,
                  divisor: float = 1) -> Tuple[int, ...]:
    wf = shape[-1] * scale
    hf = shape[-2] * scale
    if divisor > 1:
        w = divisor * ((round(wf) + divisor - 1) // divisor)
        h = divisor * ((round(hf) + divisor - 1) // divisor)
    else:
        w = math.ceil(wf)
        h = math.ceil(hf)
    return tuple(shape[0:-2]) + (h, w)


def _normalize_image(im: NDImage) -> dask.array.Array:
    if isinstance(im, dask.array.Array):
        return im
    return dask.array.from_array(im)


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


def _is_no_op(im: NDImage,
              scale: float,
              offset: np.ndarray,
              shape: Tuple[int, ...]):
    if math.isclose(scale, 1) and shape == im.shape:
        offset_y, offset_x = offset
        return math.isclose(offset_x, 0) and math.isclose(offset_y, 0)
    return False
