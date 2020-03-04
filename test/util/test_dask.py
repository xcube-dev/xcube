import unittest

import numpy as np

from xcube.util.dask import _NestedList
from xcube.util.dask import compute_array_from_func
from xcube.util.dask import get_chunk_sizes
from xcube.util.dask import get_chunk_slice_tuples


class DaskTest(unittest.TestCase):
    def test_from_func_8x10_3x4(self):
        self._assert_from_func(shape=(8, 10), chunks=(3, 4))

    def test_from_func_80x100_3x4(self):
        self._assert_from_func(shape=(80, 100), chunks=(3, 4))

    def test_from_func_80x100_7x10(self):
        self._assert_from_func(shape=(80, 100), chunks=(7, 10))

    def test_from_func_80x100_40x25(self):
        self._assert_from_func(shape=(80, 100), chunks=(40, 25))

    def _assert_from_func(self, shape, chunks):
        def my_func(array_shape, array_dtype, block_shape, block_slices):
            bsy, bsx = block_slices
            bh, bw = block_shape
            w = array_shape[-1]
            i0 = bsx[0]
            j0 = bsy[0]
            a = np.ndarray((bh, bw), dtype=array_dtype)
            for j in range(bh):
                for i in range(bw):
                    a[j, i] = 0.1 * (i0 + i + (j0 + j) * w)
            return a

        a = compute_array_from_func(my_func,
                                    shape,
                                    chunks,
                                    np.float64,
                                    ctx_arg_names=['shape', 'dtype', 'block_shape', 'block_slices'])

        self.assertIsNotNone(a)
        self.assertEqual(shape, a.shape)
        self.assertEqual(tuple(get_chunk_sizes(shape, chunks)), a.chunks)
        self.assertEqual(np.float64, a.dtype)

        h, w = shape
        n = w * h

        # Compute result
        actual = np.array(a)
        expected = (0.1 * np.linspace(0, n - 1, n, dtype=np.float64)).reshape(shape)
        np.testing.assert_almost_equal(actual, expected)

    def test_get_chunk_sizes(self):
        chunks = get_chunk_sizes((100, 100, 40), (30, 25, 50))
        self.assertEqual([(30, 30, 30, 10),
                          (25, 25, 25, 25),
                          (40,)],
                         list(chunks))

        chunks = get_chunk_sizes((100, 100, 40), (100, 100, 40))
        self.assertEqual([(100,),
                          (100,),
                          (40,)],
                         list(chunks))

    def test_get_chunk_slices_tuples(self):
        chunk_slices_tuples = get_chunk_slice_tuples([(30, 30, 30, 10),
                                                      (25, 25, 25, 25),
                                                      (40,)])
        self.assertEqual([(slice(0, 30),
                           slice(30, 60),
                           slice(60, 90),
                           slice(90, 100)),
                          (slice(0, 25),
                           slice(25, 50),
                           slice(50, 75),
                           slice(75, 100)),
                          (slice(0, 40),)],
                         list(chunk_slices_tuples))


class NestedListTest(unittest.TestCase):
    def test_len(self):
        nl = _NestedList((4,))
        self.assertEqual(4, len(nl))

        nl = _NestedList((3, 4))
        self.assertEqual(3, len(nl))

        nl = _NestedList((2, 3, 4))
        self.assertEqual(2, len(nl))

    def test_shape(self):
        nl = _NestedList([4])
        self.assertEqual((4,), nl.shape)

        nl = _NestedList([3, 4])
        self.assertEqual((3, 4), nl.shape)

        nl = _NestedList([2, 3, 4])
        self.assertEqual((2, 3, 4), nl.shape)

    def test_data(self):
        nl = _NestedList((4,))
        self.assertEqual([None, None, None, None], nl.data)

        nl = _NestedList((3, 4))
        self.assertEqual([[None, None, None, None],
                          [None, None, None, None],
                          [None, None, None, None]], nl.data)

        nl = _NestedList((2, 3, 4))
        self.assertEqual([[[None, None, None, None],
                           [None, None, None, None],
                           [None, None, None, None]],
                          [[None, None, None, None],
                           [None, None, None, None],
                           [None, None, None, None]]], nl.data)

    def test_setget(self):
        nl = _NestedList((4,), fill_value=0)
        nl[1] = 2
        self.assertEqual([0, 2, 0, 0], nl.data)
        self.assertEqual(0, nl.data[0])
        self.assertEqual(2, nl.data[1])
        self.assertEqual(0, nl.data[2])
        nl[2:4] = [6, 7]
        self.assertEqual([0, 2, 6, 7], nl.data)

        nl = _NestedList((4, 3), fill_value=True)
        nl[2, 1] = False
        self.assertEqual([[True, True, True],
                          [True, True, True],
                          [True, False, True],
                          [True, True, True]], nl.data)
        nl[0, 0:2] = [False, False]
        self.assertEqual([[False, False, True],
                          [True, True, True],
                          [True, False, True],
                          [True, True, True]], nl.data)
