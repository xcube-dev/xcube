import unittest

import xarray as xr

from test.sampledata import new_test_dataset
from xcube.api.api import XCubeDatasetAccessor


class XCubeDatasetAccessorTest(unittest.TestCase):
    # noinspection PyMethodMayBeStatic
    def test_init(self):
        XCubeDatasetAccessor(xr.Dataset())

    def test_installed(self):
        self.assertTrue(hasattr(xr.Dataset, "xcube"))
        ds = xr.Dataset()
        self.assertTrue(hasattr(ds, "xcube"))

    def test_new(self):
        self.assertIsInstance(xr.Dataset.xcube.new(), xr.Dataset)

    def test_select_vars(self):
        self.assertIsInstance(xr.Dataset().xcube.select_vars(), xr.Dataset)

    def test_dump(self):
        self.assertIsInstance(xr.Dataset().xcube.dump(), str)

    def test_vars_to_dim(self):
        dataset = new_test_dataset(["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-04", "2010-01-05"],
                                   precipitation=0.4, temperature=275.2)

        self.assertIsInstance(dataset.xcube.vars_to_dim(), xr.Dataset)

    def test_levels(self):
        dataset = new_test_dataset(["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-04", "2010-01-05"],
                                   precipitation=0.4, temperature=275.2)
        levels = dataset.xcube.levels(spatial_tile_shape=(45, 45))
        self.assertIsInstance(levels, list)
        self.assertEqual(3, len(levels))
        self.assertTrue(all(isinstance(level, xr.Dataset) for level in levels))
        self.assertTrue(all("precipitation" in level for level in levels))
        self.assertTrue(all("temperature" in level for level in levels))
        self.assertEqual([(5, 180, 360), (5, 90, 180), (5, 45, 90)],
                         [level.precipitation.shape for level in levels])
        self.assertEqual([((1, 1, 1, 1, 1), (45, 45, 45, 45), (45, 45, 45, 45, 45, 45, 45, 45)),
                          ((1, 1, 1, 1, 1), (45, 45), (45, 45, 45, 45)),
                          ((1, 1, 1, 1, 1), (45,), (45, 45))],
                         [level.precipitation.chunks for level in levels])
