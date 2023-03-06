import unittest

from xcube.core.mldataset import BaseMultiLevelDataset
from xcube.core.mldataset import CombinedMultiLevelDataset
from xcube.core.mldataset import MultiLevelDataset
from .helpers import get_test_dataset


class CombinedMultiLevelDatasetTest(unittest.TestCase):
    def test_it(self):
        ml_ds_1 = BaseMultiLevelDataset(
            get_test_dataset(('noise_1', 'noise_2'))
        )
        ml_ds_2 = BaseMultiLevelDataset(
            get_test_dataset(('noise_3', 'noise_4'))
        )
        ml_ds_3 = BaseMultiLevelDataset(
            get_test_dataset(('noise_5', 'noise_6'))
        )

        ml_ds = CombinedMultiLevelDataset([ml_ds_1, ml_ds_2, ml_ds_3])
        self.assert_ml_dataset_structure_ok(ml_ds)
        ml_ds.close()

        ml_ds = CombinedMultiLevelDataset([ml_ds_1, ml_ds_2, ml_ds_3],
                                          combiner_function=None)
        self.assert_ml_dataset_structure_ok(ml_ds)
        ml_ds.close()

    def assert_ml_dataset_structure_ok(self, ml_ds: MultiLevelDataset):
        self.assertEqual(3, ml_ds.num_levels)
        self.assertEqual((180, 180), ml_ds.grid_mapping.tile_size)

        expected_var_names = {'noise_1', 'noise_2',
                              'noise_3', 'noise_4',
                              'noise_5', 'noise_6'}

        ds0 = ml_ds.get_dataset(0)
        self.assertEqual({'time': 14, 'lat': 720, 'lon': 1440, 'bnds': 2},
                         ds0.dims)
        self.assertEqual(expected_var_names, set(map(str, ds0.data_vars)))
        self.assertTrue(all(v.dims == ('time', 'lat', 'lon')
                            for v in ds0.data_vars.values()))

        ds1 = ml_ds.get_dataset(1)
        self.assertEqual({'time': 14, 'lat': 360, 'lon': 720},
                         ds1.dims)
        self.assertEqual(expected_var_names, set(map(str, ds1.data_vars)))
        self.assertTrue(all(v.dims == ('time', 'lat', 'lon')
                            for v in ds1.data_vars.values()))

        ds2 = ml_ds.get_dataset(2)
        self.assertEqual({'time': 14, 'lat': 180, 'lon': 360},
                         ds2.dims)
        self.assertEqual(expected_var_names, set(map(str, ds2.data_vars)))
        self.assertTrue(all(v.dims == ('time', 'lat', 'lon')
                            for v in ds2.data_vars.values()))

        self.assertEqual([ds0, ds1, ds2], ml_ds.datasets)
