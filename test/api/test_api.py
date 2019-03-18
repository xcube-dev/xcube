import unittest

import xarray as xr

from test.sampledata import new_test_dataset
from xcube.api.api import XCubeAPI


class XCubeAPITest(unittest.TestCase):
    # noinspection PyMethodMayBeStatic
    def test_init(self):
        XCubeAPI(xr.Dataset())

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
