# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

import xarray as xr

from xcube.core.mldataset import BaseMultiLevelDataset, IdentityMultiLevelDataset

from .helpers import get_test_dataset


class IdentityMultiLevelDatasetTest(unittest.TestCase):
    def test_it(self):
        mlds = BaseMultiLevelDataset(get_test_dataset())
        imlds = IdentityMultiLevelDataset(mlds)
        self.assertEqual(imlds.num_levels, mlds.num_levels)
        self.assertEqual(
            set(imlds.base_dataset.data_vars), set(mlds.base_dataset.data_vars)
        )
        self.assertEqual(
            set(imlds.base_dataset.data_vars), set(mlds.base_dataset.data_vars)
        )
        xr.testing.assert_allclose(imlds.base_dataset.noise, mlds.base_dataset.noise)
