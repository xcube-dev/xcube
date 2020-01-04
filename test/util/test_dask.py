import unittest

import numpy as np

from xcube.util.dask import ChunkContext, compute_array_from_func, get_chunk_sizes, get_chunk_slice_tuples


class DaskTest(unittest.TestCase):
    def test_from_func(self):
        def my_func(context: ChunkContext):
            csy, csx = context.chunk_slices
            ch, cw = context.chunk_shape
            w = context.array_shape[-1]
            a = np.ndarray((ch, cw), dtype=context.dtype)
            for j in range(ch):
                for i in range(cw):
                    a[j, i] = 0.1 * ((csy.start + j) * w + csx.start + i)
            return a

        a = compute_array_from_func(my_func, (8, 10), (3, 4), np.float64)

        self.assertIsNotNone(a)
        self.assertEqual((8, 10), a.shape)
        self.assertEqual(((3, 3, 2), (4, 4, 2)), a.chunks)
        self.assertEqual(np.float64, a.dtype)

        # Compute result
        actual = np.array(a)
        expected = (0.1 * np.linspace(0, 8 * 10 - 1, 8 * 10, dtype=np.float64)).reshape((8, 10))
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
