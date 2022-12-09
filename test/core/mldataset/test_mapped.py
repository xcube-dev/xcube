import unittest

import xarray as xr

from xcube.core.mldataset import BaseMultiLevelDataset
from xcube.core.mldataset import MappedMultiLevelDataset
from .helpers import get_test_dataset


class IdentityMultiLevelDatasetTest(unittest.TestCase):
    def test_it(self):
        def map_ds(ds):
            return xr.Dataset(data_vars=dict(noise=ds.noise * 2))

        mlds = BaseMultiLevelDataset(get_test_dataset())
        mmlds = MappedMultiLevelDataset(mlds, map_ds)
        self.assertEqual(mmlds.num_levels, mlds.num_levels)
        self.assertEqual(set(mmlds.base_dataset.data_vars),
                         set(mlds.base_dataset.data_vars))
        self.assertEqual(set(mmlds.base_dataset.data_vars),
                         set(mlds.base_dataset.data_vars))
        xr.testing.assert_allclose(mmlds.base_dataset.noise,
                                   2 * mlds.base_dataset.noise)
