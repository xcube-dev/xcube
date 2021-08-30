import unittest

import xarray as xr

# noinspection PyUnresolvedReferences
import xcube.core.xarray
from xcube.core.gridmapping import GridMapping
from xcube.core.new import new_cube


class XCubeDatasetAccessorTest(unittest.TestCase):

    def test_installed(self):
        self.assertTrue(hasattr(xr.Dataset, "xcube"))
        ds = xr.Dataset()
        self.assertTrue(hasattr(ds, "xcube"))

    def test_cube_and_gm(self):
        dataset = new_cube(variables=dict(a=9, b=0.2))

        cube = dataset.xcube.cube
        self.assertIsInstance(cube, xr.Dataset)
        self.assertEqual(set(dataset.data_vars), set(cube.data_vars))
        gm = dataset.xcube.gm
        self.assertIsInstance(gm, GridMapping)

        dataset = xr.Dataset(dict(a=9, b=0.2))

        cube = dataset.xcube.cube
        self.assertIsInstance(cube, xr.Dataset)
        self.assertEqual(set(), set(cube.data_vars))
        gm = dataset.xcube.gm
        self.assertIsNone(gm)
