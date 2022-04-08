import unittest
import warnings
from typing import Tuple

import numpy as np
import xarray as xr
from zarr.storage import MemoryStore

from xcube.core.chunkstore import ChunkStore
from xcube.core.chunkstore import LoggingStore
from xcube.core.chunkstore import MutableLoggingStore


class ChunkStoreTest(unittest.TestCase):
    def test_chunk_store(self):
        self._test_chunk_store(trace_store_calls=False)

    def test_chunk_store_with_tracing(self):
        self._test_chunk_store(trace_store_calls=True)

    def _test_chunk_store(self, trace_store_calls: bool):
        index_var = gen_index_var(dims=('time', 'lat', 'lon'),
                                  shape=(4, 8, 16),
                                  chunks=(2, 4, 8),
                                  trace_store_calls=trace_store_calls)
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


def gen_index_var(dims, shape, chunks, trace_store_calls: bool = False):
    # noinspection PyUnusedLocal
    def get_chunk(cube_store: ChunkStore, name: str,
                  index: Tuple[int, ...]) -> bytes:
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

    store = ChunkStore(dims, shape, chunks,
                       trace_store_calls=trace_store_calls)
    store.add_lazy_array('__index_var__', '<u8', get_chunk=get_chunk)

    ds = xr.open_zarr(store)
    return ds.__index_var__


class LoggingStoreTest(unittest.TestCase):
    def test_immutable_new(self):
        self.assertReadOk(LoggingStore.new(self.original_store))

    def test_mutable_new(self):
        self.assertWriteOk(LoggingStore.new(self.original_store))

    def test_immutable(self):
        self.assertReadOk(LoggingStore(self.original_store))

    def test_mutable(self):
        self.assertWriteOk(MutableLoggingStore(self.original_store))

    def setUp(self) -> None:
        self.zattrs_value = bytes()
        self.original_store = MemoryStore()
        self.original_store.update({'chl/.zattrs': self.zattrs_value})

    def assertReadOk(self, logging_store: LoggingStore):
        # noinspection PyUnresolvedReferences
        self.assertEqual(['.zattrs'],
                         logging_store.listdir('chl'))
        # noinspection PyUnresolvedReferences
        self.assertEqual(0,
                         logging_store.getsize('chl'))
        self.assertEqual({'chl/.zattrs'},
                         set(logging_store.keys()))
        self.assertEqual(['chl/.zattrs'],
                         list(iter(logging_store)))
        self.assertTrue('chl/.zattrs' in logging_store)
        self.assertEqual(1,
                         len(logging_store))
        self.assertEqual(self.zattrs_value,
                         logging_store.get('chl/.zattrs'))
        # assert original_store not changed
        self.assertEqual({'chl/.zattrs'},
                         set(self.original_store.keys()))

    def assertWriteOk(self, logging_store: MutableLoggingStore):
        zarray_value = bytes()
        logging_store['chl/.zarray'] = zarray_value
        self.assertEqual({'chl/.zattrs',
                          'chl/.zarray'},
                         set(self.original_store.keys()))
        del logging_store['chl/.zarray']
        self.assertEqual({'chl/.zattrs'},
                         set(self.original_store.keys()))
