import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from test.test_data import create_highroc_dataset
from xcube.snap.mask import mask_dataset

nan = np.nan


class MaskDatasetTest(unittest.TestCase):
    def test_mask_dataset(self):
        dataset = create_highroc_dataset()
        masked_dataset, mask_sets = mask_dataset(dataset, errors='raise')
        self.assertIsNotNone(masked_dataset)
        self.assertIsNotNone(mask_sets)
        self.assertIsNotNone(len(mask_sets), 4)

        expected_conc_chl = np.array([[7., 11., nan, 5.],
                                      [5., nan, 2., nan],
                                      [nan, 6., 20., 17.]], dtype=np.float32)
        self.assertIn('conc_chl', masked_dataset)
        # print(masked_dataset.conc_chl)
        self.assertEqual(masked_dataset.conc_chl.shape, (3, 4))
        self.assertEqual(masked_dataset.conc_chl.dtype, np.float32)
        assert_array_almost_equal(masked_dataset.conc_chl, expected_conc_chl)

        expected_c2rcc_flags = np.array([[1, 1, 1, 1],
                                         [1, 4, 1, 2],
                                         [8, 1, 1, 1]], dtype=np.uint8)
        self.assertIn('c2rcc_flags', masked_dataset)
        # print(masked_dataset.c2rcc_flags)
        self.assertEqual(masked_dataset.c2rcc_flags.shape, (3, 4))
        self.assertEqual(masked_dataset.c2rcc_flags.dtype, np.uint8)
        assert_array_almost_equal(masked_dataset.c2rcc_flags, expected_c2rcc_flags)
