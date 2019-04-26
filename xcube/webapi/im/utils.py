# The MIT License (MIT)
# Copyright (c) 2018 by the xcube development team and contributors
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

__author__ = "Norman Fomferra (Brockmann Consult GmbH)"

import numpy as np


def aggregate_ndarray_first(a1, a2, a3, a4):
    return a1


def aggregate_ndarray_min(a1, a2, a3, a4):
    a = np.fmin(a1, a2)
    a = np.fmin(a, a3, out=a)
    a = np.fmin(a, a4, out=a)
    return a


def aggregate_ndarray_max(a1, a2, a3, a4):
    a = np.fmax(a1, a2)
    a = np.fmax(a, a3, out=a)
    a = np.fmax(a, a4, out=a)
    return a


def aggregate_ndarray_sum(a1, a2, a3, a4):
    return a1 + a2 + a3 + a4


def aggregate_ndarray_mean(a1, a2, a3, a4):
    return (a1 + a2 + a3 + a4) / 4.


def downsample_ndarray(a, aggregator=aggregate_ndarray_mean):
    if aggregator is aggregate_ndarray_first:
        # Optimization
        return a[..., 0::2, 0::2]
    else:
        a1 = a[..., 0::2, 0::2]
        a2 = a[..., 0::2, 1::2]
        a3 = a[..., 1::2, 0::2]
        a4 = a[..., 1::2, 1::2]
        return aggregator(a1, a2, a3, a4)


def get_chunk_size(array):
    chunk_size = None
    try:
        # xarray DataArray with dask, returns the size of each individual tile
        chunk_size = array.chunks
        if chunk_size:
            chunk_size = tuple([c[0] if isinstance(c, tuple) else c for c in chunk_size])
    except Exception:
        pass
    if not chunk_size:
        try:
            # netcdf 4
            chunk_size = array.encoding['chunksizes']
        except Exception:
            pass
    return chunk_size
