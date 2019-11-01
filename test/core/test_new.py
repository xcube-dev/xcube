import unittest

import numpy as np

from xcube.core.new import new_cube


class NewCubeTest(unittest.TestCase):

    def test_new_cube_with_bounds(self):
        cube = new_cube()
        self.assertEqual({'lon': 360, 'lat': 180, 'time': 5, 'bnds': 2}, cube.dims)
        self.assertEqual(-179.5, float(cube.lon[0]))
        self.assertEqual(179.5, float(cube.lon[-1]))
        self.assertEqual(-89.5, float(cube.lat[0]))
        self.assertEqual(89.5, float(cube.lat[-1]))
        self.assertEqual(-180., float(cube.lon_bnds[0, 0]))
        self.assertEqual(-179., float(cube.lon_bnds[0, 1]))
        self.assertEqual(179., float(cube.lon_bnds[-1, 0]))
        self.assertEqual(180., float(cube.lon_bnds[-1, 1]))
        self.assertEqual(-90., float(cube.lat_bnds[0, 0]))
        self.assertEqual(-89., float(cube.lat_bnds[0, 1]))
        self.assertEqual(89., float(cube.lat_bnds[-1, 0]))
        self.assertEqual(90., float(cube.lat_bnds[-1, 1]))

    def test_new_cube_without_bounds(self):
        cube = new_cube(drop_bounds=True)
        self.assertEqual({'lon': 360, 'lat': 180, 'time': 5}, cube.dims)

    def test_new_cube_with_const_vars(self):
        cube = new_cube(variables=dict(sst=274.4, chl=10.31))
        self.assertIn('sst', cube)
        self.assertIn('chl', cube)
        import numpy as np
        np.testing.assert_almost_equal(cube.sst.values, np.full((5, 180, 360), 274.4))
        np.testing.assert_almost_equal(cube.chl.values, np.full((5, 180, 360), 10.31))

    def test_new_cube_with_func_vars(self):
        def sst_func(t, y, x):
            return 274.4 + t * 0.5

        def chl_func(t, y, x):
            return 10.31 + y * 0.1

        def aot_func(t, y, x):
            return 0.88 + x * 0.05

        cube = new_cube(width=16, height=8, variables=dict(sst=sst_func, chl=chl_func, aot=aot_func))

        self.assertIn('sst', cube)
        self.assertIn('chl', cube)
        self.assertIn('aot', cube)

        np.testing.assert_almost_equal(cube.sst[0, :, :].values, np.full((8, 16), 274.4))
        np.testing.assert_almost_equal(cube.chl[:, 0, :].values, np.full((5, 16), 10.31))
        np.testing.assert_almost_equal(cube.aot[:, :, 0].values, np.full((5, 8), 0.88))

        np.testing.assert_almost_equal(cube.sst.isel(lon=3, lat=2).values,
                                       np.array([274.4, 274.9, 275.4, 275.9, 276.4]))
        np.testing.assert_almost_equal(cube.chl.isel(time=3, lon=4).values,
                                       np.array([10.31, 10.41, 10.51, 10.61, 10.71, 10.81, 10.91, 11.01]))
        np.testing.assert_almost_equal(cube.aot.isel(time=1, lat=2).values,
                                       np.array([0.88, 0.93, 0.98, 1.03, 1.08, 1.13, 1.18, 1.23,
                                                 1.28, 1.33, 1.38, 1.43, 1.48, 1.53, 1.58, 1.63]))
