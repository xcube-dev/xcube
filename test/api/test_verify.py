import unittest

import numpy as np
import xarray as xr

from xcube.api.new import new_cube
from xcube.api.verify import verify_cube, assert_cube


class AssertAndVerifyCubeTest(unittest.TestCase):

    def test_assert_cube(self):
        cube = new_cube(variables=dict(precipitation=0.5))

        cube_2 = assert_cube(cube)
        self.assertIs(cube_2, cube)

        cube["chl"] = xr.DataArray(np.random.rand(cube.dims["lat"], cube.dims["lon"]),
                                   dims=("lat", "lon"),
                                   coords=dict(lat=cube.lat, lon=cube.lon))
        with self.assertRaises(ValueError) as cm:
            assert_cube(cube)
        self.assertEqual("Dataset is not a valid data cube, because:\n"
                         "- dimensions of data variable 'chl' must be"
                         " ('time', ..., 'lat', 'lon'), but were ('lat', 'lon') for 'chl';\n"
                         "- dimensions of all data variables must be same,"
                         " but found ('time', 'lat', 'lon') for 'precipitation'"
                         " and ('lat', 'lon') for 'chl'.",
                         f"{cm.exception}")

    def test_verify_cube(self):
        cube = new_cube()
        self.assertEqual([], verify_cube(cube))
        ds = cube.drop("time")
        self.assertEqual(["missing coordinate variable 'time'"], verify_cube(ds))
        ds = ds.drop("lat")
        self.assertEqual(["missing coordinate variable 'time'",
                          "missing coordinate variable 'lat'"], verify_cube(ds))
