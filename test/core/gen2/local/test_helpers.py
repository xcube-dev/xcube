import unittest

from xcube.core.gen2.local.helpers import is_empty_cube
from xcube.core.gen2.local.helpers import strip_cube
from xcube.core.new import new_cube


class HelpersTest(unittest.TestCase):
    def test_is_empty_cube(self):
        cube = new_cube()
        self.assertEqual(True, is_empty_cube(cube))

        cube = new_cube(variables=dict(chl=0.6, tsm=0.9, flags=16))
        self.assertEqual(False, is_empty_cube(cube))

    def test_strip_cube(self):
        cube = new_cube(variables=dict(chl=0.6, tsm=0.9, flags=16))
        self.assertIs(cube, strip_cube(cube))

        cube = new_cube(variables=dict(chl=0.6, tsm=0.9, flags=16))
        cube_subset = cube.sel(time=slice('1990-01-01', '1991-01-01'))
        stripped_cube = strip_cube(cube_subset)
        self.assertEqual(set(), set(stripped_cube.data_vars))

        cube = new_cube(variables=dict(chl=0.6, tsm=0.9, flags=16))
        cube_subset = cube.sel(lat=0, lon=0, method='nearest')
        stripped_cube = strip_cube(cube_subset)
        self.assertEqual(set(), set(stripped_cube.data_vars))
