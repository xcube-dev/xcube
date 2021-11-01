import unittest

import numpy as np
import xarray as xr

from xcube.core.new import new_cube
from xcube.core.verify import verify_cube, assert_cube


class AssertAndVerifyCubeTest(unittest.TestCase):

    def test_assert_cube_ok(self):
        cube = new_cube(variables=dict(precipitation=0.5))
        self.assertIs(cube, assert_cube(cube))

    def test_assert_cube_without_bounds(self):
        cube = new_cube(variables=dict(precipitation=0.5), drop_bounds=True)
        self.assertIs(cube, assert_cube(cube))

    def test_assert_cube_illegal_coord_var(self):
        cube = new_cube(variables=dict(precipitation=0.5))
        cube = cube.assign_coords(lat=xr.DataArray(np.outer(cube.lat, np.ones(cube.lon.size)),
                                                   dims=("y", "x")),
                                  lon=xr.DataArray(np.outer(np.ones(cube.lat.size), cube.lon),
                                                   dims=("y", "x")))
        with self.assertRaises(ValueError) as cm:
            assert_cube(cube)
        self.assertEqual("Dataset is not a valid xcube dataset, because:\n"
                         "- missing spatial x,y coordinate variables.",
                         f"{cm.exception}")

    def test_assert_cube_illegal_coord_bounds_var(self):
        cube = new_cube(variables=dict(precipitation=0.5))
        lat_bnds = np.zeros((cube.time.size, cube.lat.size, 2))
        lon_bnds = np.zeros((cube.time.size, cube.lon.size, 2), dtype=np.float16)
        lat_bnds[:, :, :] = cube.lat_bnds
        lon_bnds[:, :, :] = cube.lon_bnds
        cube = cube.assign_coords(lat_bnds=xr.DataArray(lat_bnds, dims=("time", "lat", "bnds")),
                                  lon_bnds=xr.DataArray(lon_bnds, dims=("time", "lon", "bnds")))
        with self.assertRaises(ValueError) as cm:
            assert_cube(cube)
        self.assertEqual("Dataset is not a valid xcube dataset, because:\n"
                         "- bounds coordinate variable 'lon_bnds' must have dimensions ('lon', <bounds_dim>);\n"
                         "- shape of bounds coordinate variable 'lon_bnds' must be (360, 2) but was (5, 360, 2);\n"
                         "- type of bounds coordinate variable 'lon_bnds' must be dtype('float64') but was dtype('float16');\n"
                         "- bounds coordinate variable 'lat_bnds' must have dimensions ('lat', <bounds_dim>);\n"
                         "- shape of bounds coordinate variable 'lat_bnds' must be (180, 2) but was (5, 180, 2).",
                         f"{cm.exception}")

    def test_assert_cube_illegal_data_var(self):
        cube = new_cube(variables=dict(precipitation=0.5))
        shape = cube.dims["lat"], cube.dims["lon"]
        cube["chl"] = xr.DataArray(np.random.rand(*shape),
                                   dims=("lat", "lon"),
                                   coords=dict(lat=cube.lat, lon=cube.lon))
        with self.assertRaises(ValueError) as cm:
            assert_cube(cube)
        self.assertEqual("Dataset is not a valid xcube dataset, because:\n"
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
        self.assertEqual(["missing time coordinate variable"], verify_cube(ds))
        ds = ds.drop("lat")
        self.assertEqual(["missing spatial x,y coordinate variables",
                          "missing time coordinate variable"], verify_cube(ds))

    def test_verify_cube_coord_equidistance(self):
        ds = new_cube()
        ds['lat'] = ds['lat'] + np.random.rand(len(ds['lat']))
        ds['lat_bnds'][:, 0] = ds['lat_bnds'][:, 0] + np.random.rand(len(ds['lat_bnds'][:, 0]))
        ds['lat_bnds'][:, 1] = ds['lat_bnds'][:, 1] + np.random.rand(len(ds['lat_bnds'][:, 1]))

        self.assertEqual(["coordinate variable 'lat' is not equidistant",
                          "coordinate variable 'lat_bnds' is not equidistant"],
                         verify_cube(ds))

    def test_verify_cube_coord_equidistance_reverse_lat(self):
        ds = new_cube()
        ds['lat'] = np.flip(ds.lat)
        ds['lat_bnds'] = xr.DataArray(np.flip(ds.lat_bnds.values),
                                      dims=['lat', 'bnds'])

        ds['lon'] = np.roll(ds.lon, 90)
        ds['lon_bnds'] = xr.DataArray(np.roll(ds.lon_bnds.values, 180),
                                      dims=['lon', 'bnds'])

        self.assertEqual([], verify_cube(ds))
