import unittest

from xcube.api import new_cube


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
