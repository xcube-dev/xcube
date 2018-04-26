import os
import unittest

import numpy as np
import xarray as xr
from numpy.testing import assert_array_almost_equal

from test.reproject_test import HIGHROC_NC
from xcube.mask import LazyMaskSet, mask_dataset


class LazyMaskSetTest(unittest.TestCase):
    def test_mask_flags_highroc(self):
        if not os.path.isfile(HIGHROC_NC):
            print('warning: test_reproject_xarray() not executed')
        dataset = xr.open_dataset(HIGHROC_NC, decode_cf=True, decode_coords=False)
        mask_sets = LazyMaskSet.get_mask_sets(dataset)
        for mask_name, mask_set in mask_sets.items():
            print(mask_name, mask_set)

        mask_dataset(dataset, errors='warn')

    def test_mask_flags(self):
        nan = float('nan')

        conc_chl = np.array([[7, 11, nan, 5],
                             [5, 10, 2, 21],
                             [16, 6, 20, 17]], dtype=np.float32)

        c2rcc_flags = np.array([[1, 1, 1, 1],
                                [1, 4, 1, 2],
                                [8, 1, 1, 1]], dtype=np.uint8)

        lon = np.array([[8, 9, 10, 11],
                        [8, 9, 10, 11],
                        [8, 9, 10, 11]], dtype=np.float32)

        lat = np.array([[56, 56, 56, 53],
                        [55, 55, 55, 55],
                        [54, 54, 54, 54]], dtype=np.float32)

        dataset = xr.Dataset(dict(
            conc_chl=(('y', 'x'), conc_chl, dict(
                long_name="Chlorophylll concentration",
                units="mg m^-3",
                _FillValue=nan,
                valid_pixel_expression="c2rcc_flags.F1",
            )),
            c2rcc_flags=(('y', 'x'), c2rcc_flags, dict(
                long_name="C2RCC quality flags",
                _Unsigned="true",
                flag_meanings="F1 F2 F3 F4",
                flag_masks=np.array([1, 2, 4, 8], np.int32),
                flag_coding_name="c2rcc_flags",
                flag_descriptions="D1 D2 D3 D4",
            )),
            lon=(('y', 'x'), lon, dict(
                long_name="longitude",
                units="degrees east",
            )),
            lat=(('y', 'x'), lat, dict(
                long_name="latitude",
                units="degrees north",
            )),
        ))

        mask_sets = LazyMaskSet.get_mask_sets(dataset)
        self.assertIsNotNone(mask_sets)
        self.assertEqual(len(mask_sets), 1)
        self.assertIn('c2rcc_flags', mask_sets)
        mask_set = mask_sets['c2rcc_flags']
        self.assertIsInstance(mask_set, LazyMaskSet)

    def test_lazy_mask(self):

        c2rcc_flags = np.array([[1, 1, 1, 1],
                                [1, 4, 1, 2],
                                [8, 1, 1, 1]], dtype=np.uint8)

        flag_var = xr.DataArray(c2rcc_flags, dims=('y', 'x'), name='c2rcc_flags', attrs=dict(
            long_name="C2RCC quality flags",
            _Unsigned="true",
            flag_meanings="F1 F2 F3 F4",
            flag_masks=np.array([1, 2, 4, 8], np.int32),
            flag_coding_name="c2rcc_flags",
            flag_descriptions="D1 D2 D3 D4",
        ))

        lazy_mask_set = LazyMaskSet(flag_var)
        self.assertEqual(str(lazy_mask_set), 'c2rcc_flags(F1=1, F2=2, F3=4, F4=8)')

        mask_f1 = lazy_mask_set.F1
        self.assertIs(mask_f1, lazy_mask_set.F1)
        mask_f2 = lazy_mask_set.F2
        self.assertIs(mask_f2, lazy_mask_set.F2)
        mask_f3 = lazy_mask_set.F3
        self.assertIs(mask_f3, lazy_mask_set.F3)
        mask_f4 = lazy_mask_set.F4
        self.assertIs(mask_f4, lazy_mask_set.F4)

        validation_data = ((0, 'F1', mask_f1, np.array([[0, 0, 0, 0],
                                                        [0, 1, 0, 1],
                                                        [1, 0, 0, 0]],
                                                       dtype=np.uint8)),
                           (1, 'F2', mask_f2, np.array([[1, 1, 1, 1],
                                                        [1, 1, 1, 0],
                                                        [1, 1, 1, 1]],
                                                       dtype=np.uint8)),
                           (2, 'F3', mask_f3, np.array([[1, 1, 1, 1],
                                                        [1, 0, 1, 1],
                                                        [1, 1, 1, 1]],
                                                       dtype=np.uint8)),
                           (3, 'F4', mask_f4, np.array([[1, 1, 1, 1],
                                                        [1, 1, 1, 1],
                                                        [0, 1, 1, 1]],
                                                       dtype=np.uint8)))

        for index, name, mask, data in validation_data:
            self.assertIs(mask, lazy_mask_set[index])
            self.assertIs(mask, lazy_mask_set[name])
            assert_array_almost_equal(mask.values, data)
