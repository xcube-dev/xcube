import unittest

import xarray as xr

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
