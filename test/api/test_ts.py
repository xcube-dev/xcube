import unittest

import numpy as np
import xarray as xr

from xcube.api import new_cube
from xcube.api.ts import get_time_series


class TsTest(unittest.TestCase):

    def test_point(self):
        ts_ds = get_time_series(self.cube, dict(type='Point', coordinates=[20.0, 10.0]))
        self.assert_dataset_ok(ts_ds, 1, True, False, False, True, False, False)

    def test_polygon(self):
        ts_ds = get_time_series(self.cube,
                                dict(type='Polygon',
                                     coordinates=[[[20.0, 10.0],
                                                   [20.0, 20.0],
                                                   [10.0, 20.0],
                                                   [10.0, 10.0],
                                                   [20.0, 10.0]]]))
        self.assert_dataset_ok(ts_ds, 100, True, False, False, True, False, False)

    def test_polygon_incl_count_stdev(self):
        ts_ds = get_time_series(self.cube,
                                dict(type='Polygon',
                                     coordinates=[[[20.0, 10.0],
                                                   [20.0, 20.0],
                                                   [10.0, 20.0],
                                                   [10.0, 10.0],
                                                   [20.0, 10.0]]]),
                                include_count=True,
                                include_stdev=True)
        self.assert_dataset_ok(ts_ds, 100, True, True, True, True, True, True)

    def test_polygon_incl_count(self):
        ts_ds = get_time_series(self.cube,
                                dict(type='Polygon',
                                     coordinates=[[[20.0, 10.0],
                                                   [20.0, 20.0],
                                                   [10.0, 20.0],
                                                   [10.0, 10.0],
                                                   [20.0, 10.0]]]),
                                include_count=True)
        self.assert_dataset_ok(ts_ds, 100, True, True, False, True, True, False)

    def test_polygon_incl_stdev_var_subs(self):
        ts_ds = get_time_series(self.cube,
                                dict(type='Polygon',
                                     coordinates=[[[20.0, 10.0],
                                                   [20.0, 20.0],
                                                   [10.0, 20.0],
                                                   [10.0, 10.0],
                                                   [20.0, 10.0]]]),
                                var_names=['B'],
                                include_stdev=True)
        self.assert_dataset_ok(ts_ds, 100, False, False, False, True, False, True)

    def test_polygon_using_groupby(self):
        ts_ds = get_time_series(self.cube,
                                dict(type='Polygon',
                                     coordinates=[[[20.0, 10.0],
                                                   [20.0, 20.0],
                                                   [10.0, 20.0],
                                                   [10.0, 10.0],
                                                   [20.0, 10.0]]]),
                                include_count=True,
                                include_stdev=True,
                                use_groupby=True)
        self.assert_dataset_ok(ts_ds, 100, True, True, True, True, True, True)

    def test_no_vars(self):
        ts_ds = get_time_series(self.cube,
                                dict(type='Polygon',
                                     coordinates=[[[20.0, 10.0],
                                                   [20.0, 20.0],
                                                   [10.0, 20.0],
                                                   [10.0, 10.0],
                                                   [20.0, 10.0]]]),
                                var_names=[])
        self.assertIsNone(ts_ds)

    def setUp(self):
        shape = 25, 180, 360
        dims = 'time', 'lat', 'lon'
        self.ts_a = np.linspace(1, 25, 25)
        self.ts_a_count = np.array(25 * [100])
        self.ts_a_stdev = np.array(25 * [0.0])
        self.ts_b = np.linspace(0, 1, 25)
        self.ts_b_count = np.array(25 * [100])
        self.ts_b_stdev = np.array(25 * [0.0])
        self.cube = new_cube(time_periods=25,
                             variables=dict(
                                 A=xr.DataArray(np.broadcast_to(self.ts_a.reshape(25, 1, 1), shape), dims=dims),
                                 B=xr.DataArray(np.broadcast_to(self.ts_b.reshape(25, 1, 1), shape), dims=dims)
                             ))
        self.cube = self.cube.chunk(chunks=dict(time=1, lat=180, lon=180))

    def assert_dataset_ok(self, ts_ds,
                          expected_max_number_of_observations=0,
                          expected_a=False,
                          expected_a_count=False,
                          expected_a_stdev=False,
                          expected_b=False,
                          expected_b_count=False,
                          expected_b_stdev=False):
        self.assertIsNotNone(ts_ds)
        self.assertEqual(expected_max_number_of_observations, ts_ds.attrs.get('max_number_of_observations'))
        self.assert_variable_ok(ts_ds, 'A', expected_a, self.ts_a)
        self.assert_variable_ok(ts_ds, 'A_count', expected_a_count, self.ts_a_count)
        self.assert_variable_ok(ts_ds, 'A_stdev', expected_a_stdev, self.ts_a_stdev)
        self.assert_variable_ok(ts_ds, 'B', expected_b, self.ts_b)
        self.assert_variable_ok(ts_ds, 'B_count', expected_b_count, self.ts_b_count)
        self.assert_variable_ok(ts_ds, 'B_stdev', expected_b_stdev, self.ts_b_stdev)

    def assert_variable_ok(self, ts_ds, name, expected_in, expected_values):
        if expected_in:
            self.assertIn(name, ts_ds)
            self.assertEqual(('time',), ts_ds[name].dims)
            self.assertEqual(25, ts_ds[name].size)
            np.testing.assert_almost_equal(ts_ds[name].values, expected_values)
        else:
            self.assertNotIn(name, ts_ds)
