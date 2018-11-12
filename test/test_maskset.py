import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from test.sampledata import create_highroc_dataset, create_c2rcc_flag_var, create_cmems_sst_flag_var
from xcube.maskset import MaskSet


class MaskSetTest(unittest.TestCase):
    def test_mask_set_with_flag_mask_str(self):
        flag_var = create_cmems_sst_flag_var()
        mask_set = MaskSet(flag_var)

        self.assertEqual('mask(sea=(1, None), land=(2, None), lake=(4, None), ice=(8, None))',
                         str(mask_set))

        mask_f1 = mask_set.sea
        self.assertIs(mask_f1, mask_set.sea)
        mask_f2 = mask_set.land
        self.assertIs(mask_f2, mask_set.land)
        mask_f3 = mask_set.lake
        self.assertIs(mask_f3, mask_set.lake)
        mask_f4 = mask_set.ice
        self.assertIs(mask_f4, mask_set.ice)

        validation_data = ((0, 'sea', mask_f1, np.array([[[1, 0, 0, 0],
                                                          [1, 1, 0, 0],
                                                          [1, 1, 1, 0]]],
                                                        dtype=np.uint8)),
                           (1, 'land', mask_f2, np.array([[[0, 1, 0, 0],
                                                           [0, 0, 1, 1],
                                                           [0, 0, 0, 1]]],
                                                         dtype=np.uint8)),
                           (2, 'lake', mask_f3, np.array([[[0, 0, 1, 1],
                                                           [0, 0, 0, 0],
                                                           [0, 0, 0, 0]]],
                                                         dtype=np.uint8)),
                           (3, 'ice', mask_f4, np.array([[[1, 1, 1, 0],
                                                          [1, 0, 0, 0],
                                                          [0, 0, 0, 0]]],
                                                        dtype=np.uint8)))

        for index, name, mask, data in validation_data:
            self.assertIs(mask, mask_set[index])
            self.assertIs(mask, mask_set[name])
            assert_array_almost_equal(mask.values, data, err_msg=f'{index}, {name}, {mask.name}')

    def test_mask_set_with_flag_mask_int_array(self):
        flag_var = create_c2rcc_flag_var()
        mask_set = MaskSet(flag_var)

        self.assertEqual('c2rcc_flags(F1=(1, None), F2=(2, None), F3=(4, None), F4=(8, None))',
                         str(mask_set))

        mask_f1 = mask_set.F1
        self.assertIs(mask_f1, mask_set.F1)
        mask_f2 = mask_set.F2
        self.assertIs(mask_f2, mask_set.F2)
        mask_f3 = mask_set.F3
        self.assertIs(mask_f3, mask_set.F3)
        mask_f4 = mask_set.F4
        self.assertIs(mask_f4, mask_set.F4)

        validation_data = ((0, 'F1', mask_f1, np.array([[1, 1, 1, 1],
                                                        [1, 0, 1, 0],
                                                        [0, 1, 1, 1]],
                                                       dtype=np.uint8)),
                           (1, 'F2', mask_f2, np.array([[0, 0, 0, 0],
                                                        [0, 0, 0, 1],
                                                        [0, 0, 0, 0]],
                                                       dtype=np.uint8)),
                           (2, 'F3', mask_f3, np.array([[0, 0, 0, 0],
                                                        [0, 1, 0, 0],
                                                        [0, 0, 0, 0]],
                                                       dtype=np.uint8)),
                           (3, 'F4', mask_f4, np.array([[0, 0, 0, 0],
                                                        [0, 0, 0, 0],
                                                        [1, 0, 0, 0]],
                                                       dtype=np.uint8)))

        for index, name, mask, data in validation_data:
            self.assertIs(mask, mask_set[index])
            self.assertIs(mask, mask_set[name])
            assert_array_almost_equal(mask.values, data)

    def test_get_mask_sets(self):
        dataset = create_highroc_dataset()
        mask_sets = MaskSet.get_mask_sets(dataset)
        self.assertIsNotNone(mask_sets)
        self.assertEqual(len(mask_sets), 1)
        self.assertIn('c2rcc_flags', mask_sets)
        mask_set = mask_sets['c2rcc_flags']
        self.assertIsInstance(mask_set, MaskSet)

# TODO: add tests according to http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ch03s05.html
