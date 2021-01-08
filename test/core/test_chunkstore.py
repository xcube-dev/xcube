import unittest
import warnings
from typing import Tuple

import numpy as np
import xarray as xr

from xcube.core.chunkstore import ChunkStore


class ChunkStoreTest(unittest.TestCase):

    def test_chunk_store(self):
        index_var = gen_index_var(dims=('time', 'lat', 'lon'),
                                  shape=(4, 8, 16),
                                  chunks=(2, 4, 8))
        self.assertIsNotNone(index_var)
        self.assertEqual((4, 8, 16), index_var.shape)
        self.assertEqual(((2, 2), (4, 4), (8, 8)), index_var.chunks)
        self.assertEqual(('time', 'lat', 'lon'), index_var.dims)

        visited_indexes = set()

        def index_var_ufunc(index_var_):
            if index_var_.size < 6:
                warnings.warn(f"weird variable of size {index_var_.size} received!")
                return
            nonlocal visited_indexes
            index = tuple(map(int, index_var_.ravel()[0:6]))
            visited_indexes.add(index)
            return index_var_

        result = xr.apply_ufunc(index_var_ufunc, index_var, dask='parallelized', output_dtypes=[index_var.dtype])
        self.assertIsNotNone(result)
        self.assertEqual((4, 8, 16), result.shape)
        self.assertEqual(((2, 2), (4, 4), (8, 8)), result.chunks)
        self.assertEqual(('time', 'lat', 'lon'), result.dims)

        values = result.values
        self.assertEqual((4, 8, 16), values.shape)
        np.testing.assert_array_equal(index_var.values, values)

        print(visited_indexes)

        self.assertEqual(8, len(visited_indexes))
        self.assertEqual({
            (0, 2, 0, 4, 0, 8),
            (2, 4, 0, 4, 0, 8),
            (0, 2, 4, 8, 0, 8),
            (2, 4, 4, 8, 0, 8),
            (0, 2, 0, 4, 8, 16),
            (2, 4, 0, 4, 8, 16),
            (0, 2, 4, 8, 8, 16),
            (2, 4, 4, 8, 8, 16),
        }, visited_indexes)

    def test_nan_chunks(self):
        index_var = gen_index_var_nan(dims=('time', 'lat', 'lon'),
                                      shape=(4, 8, 16),
                                      chunks=(2, 4, 8))
        self.assertIsNotNone(index_var)
        self.assertEqual((4, 8, 16), index_var.shape)
        self.assertEqual(((2, 2), (4, 4), (8, 8)), index_var.chunks)
        self.assertEqual(('time', 'lat', 'lon'), index_var.dims)
        self.assertEqual(9223372036854775808, index_var.encoding['_FillValue'])
        self.assertEqual('uint64', index_var.encoding['dtype'])
        self.assertTrue(np.isnan(index_var.values).all())


def gen_index_var(dims, shape, chunks):
    # noinspection PyUnusedLocal
    def get_chunk(cube_store: ChunkStore, name: str, index: Tuple[int, ...]) -> bytes:
        data = np.zeros(cube_store.chunks, dtype=np.uint64)
        data_view = data.ravel()
        if data_view.base is not data:
            raise ValueError('view expected')
        if data_view.size < cube_store.ndim * 2:
            raise ValueError('size too small')
        for i in range(cube_store.ndim):
            j1 = cube_store.chunks[i] * index[i]
            j2 = j1 + cube_store.chunks[i]
            data_view[2 * i] = j1
            data_view[2 * i + 1] = j2
        return data.tobytes()

    store = ChunkStore(dims, shape, chunks)
    store.add_lazy_array('__index_var__', '<u8', get_chunk=get_chunk)

    ds = xr.open_zarr(store)
    return ds.__index_var__


def gen_index_var_nan(dims, shape, chunks):
    store = ChunkStore(dims, shape, chunks)
    store.add_nan_array('__index_var__', dtype='<u8', fill_value=np.nan)

    ds = xr.open_zarr(store)
    return ds.__index_var__
