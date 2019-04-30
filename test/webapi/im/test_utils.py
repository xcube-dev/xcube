from unittest import TestCase

import numpy as np

import xcube.webapi.im.utils as utils


class ArrayTest(TestCase):
    def test_numpy_reduce(self):
        nan = np.nan
        a = np.zeros((8, 6))
        a[0::2, 0::2] = 1.1
        a[0::2, 1::2] = 2.2
        a[1::2, 0::2] = 3.3
        a[1::2, 1::2] = 4.4
        a[6, 4] = np.nan

        self.assertEqual(a.shape, (8, 6))
        np.testing.assert_equal(a, np.array([[1.1, 2.2, 1.1, 2.2, 1.1, 2.2],
                                             [3.3, 4.4, 3.3, 4.4, 3.3, 4.4],
                                             [1.1, 2.2, 1.1, 2.2, 1.1, 2.2],
                                             [3.3, 4.4, 3.3, 4.4, 3.3, 4.4],
                                             [1.1, 2.2, 1.1, 2.2, 1.1, 2.2],
                                             [3.3, 4.4, 3.3, 4.4, 3.3, 4.4],
                                             [1.1, 2.2, 1.1, 2.2, nan, 2.2],
                                             [3.3, 4.4, 3.3, 4.4, 3.3, 4.4]]))

        b = utils.downsample_ndarray(a)
        self.assertEqual(b.shape, (4, 3))
        np.testing.assert_equal(b, np.array([[2.75, 2.75, 2.75],
                                             [2.75, 2.75, 2.75],
                                             [2.75, 2.75, 2.75],
                                             [2.75, 2.75, nan]]))

        b = utils.downsample_ndarray(a, aggregator=utils.aggregate_ndarray_first)
        self.assertEqual(b.shape, (4, 3))
        np.testing.assert_equal(b, np.array([[1.1, 1.1, 1.1],
                                             [1.1, 1.1, 1.1],
                                             [1.1, 1.1, 1.1],
                                             [1.1, 1.1, nan]]))

