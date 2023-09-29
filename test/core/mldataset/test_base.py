import unittest

from xcube.constants import CRS84
from xcube.core.gridmapping import GridMapping
from xcube.core.mldataset import BaseMultiLevelDataset
from xcube.core.tilingscheme import TilingScheme
from .helpers import get_test_dataset


class BaseMultiLevelDatasetTest(unittest.TestCase):
    def test_basic_props(self):
        ds = get_test_dataset()
        ml_ds = BaseMultiLevelDataset(ds)

        self.assertIsInstance(ml_ds.ds_id, str)
        self.assertIsInstance(ml_ds.grid_mapping, GridMapping)
        self.assertIsNotNone(ml_ds.lock)
        self.assertEqual(3, ml_ds.num_levels)
        self.assertEqual({'noise': 'first'}, ml_ds.agg_methods)

    def test_resolutions(self):
        ds = get_test_dataset()
        ml_ds = BaseMultiLevelDataset(ds)

        self.assertEqual([(0.25, 0.25),
                          (0.5, 0.5),
                          (1.0, 1.0)],
                         ml_ds.resolutions)
        self.assertIs(ml_ds.resolutions, ml_ds.resolutions)

        self.assertEqual([0.25, 0.5, 1.0], ml_ds.avg_resolutions)
        self.assertIs(ml_ds.avg_resolutions, ml_ds.avg_resolutions)

    def test_get_level_for_resolution(self):
        ds = get_test_dataset()
        ml_ds = BaseMultiLevelDataset(ds)

        self.assertEqual(0, ml_ds.get_level_for_resolution(0.1))
        self.assertEqual(0, ml_ds.get_level_for_resolution(0.25))
        self.assertEqual(0, ml_ds.get_level_for_resolution(0.3))
        self.assertEqual(1, ml_ds.get_level_for_resolution(0.5))
        self.assertEqual(1, ml_ds.get_level_for_resolution(0.6))
        self.assertEqual(1, ml_ds.get_level_for_resolution(0.9))
        self.assertEqual(2, ml_ds.get_level_for_resolution(1.0))
        self.assertEqual(2, ml_ds.get_level_for_resolution(1.5))
        self.assertEqual(2, ml_ds.get_level_for_resolution(10.0))

    def test_level_datasets(self):
        ds = get_test_dataset()
        ml_ds = BaseMultiLevelDataset(ds)

        ds0 = ml_ds.get_dataset(0)
        self.assertIsNot(ds, ds0)
        self.assertEqual({'time': 14, 'lat': 720, 'lon': 1440, 'bnds': 2},
                         ds0.dims)

        ds1 = ml_ds.get_dataset(1)
        self.assertIsNot(ds, ds1)
        self.assertEqual({'time': 14, 'lat': 360, 'lon': 720},
                         ds1.dims)

        ds2 = ml_ds.get_dataset(2)
        self.assertIsNot(ds, ds2)
        self.assertEqual({'time': 14, 'lat': 180, 'lon': 360},
                         ds2.dims)

        self.assertEqual([ds0, ds1, ds2], ml_ds.datasets)

        self.assertIs(ds0, ml_ds.base_dataset)

        ml_ds.close()

    def test_arg_validation(self):
        ds = get_test_dataset()
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            BaseMultiLevelDataset('test.levels')
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            BaseMultiLevelDataset(ds, grid_mapping='crs84')

    def test_derive_tiling_scheme(self):
        ds = get_test_dataset()
        ml_ds = BaseMultiLevelDataset(ds)
        tiling_scheme = ml_ds.derive_tiling_scheme(TilingScheme.GEOGRAPHIC)
        self.assertEqual(CRS84, tiling_scheme.crs_name)
        self.assertEqual(0, tiling_scheme.min_level)
        self.assertEqual(2, tiling_scheme.max_level)
