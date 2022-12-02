import unittest

import xarray as xr

from xcube.core.mldataset import BaseMultiLevelDataset
from xcube.core.mldataset import IdentityMultiLevelDataset
from .helpers import get_test_dataset


class IdentityMultiLevelDatasetTest(unittest.TestCase):
    def test_it(self):
        mlds = BaseMultiLevelDataset(get_test_dataset())
        imlds = IdentityMultiLevelDataset(mlds)
        self.assertEqual(imlds.num_levels, mlds.num_levels)
        self.assertEqual(set(imlds.base_dataset.data_vars),
                         set(mlds.base_dataset.data_vars))
        self.assertEqual(set(imlds.base_dataset.data_vars),
                         set(mlds.base_dataset.data_vars))
        xr.testing.assert_allclose(imlds.base_dataset.noise,
                                   mlds.base_dataset.noise)
