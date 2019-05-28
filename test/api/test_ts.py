import unittest

import numpy as np
import xarray as xr

from xcube.api import new_cube
from xcube.api.ts import get_time_series_for_point, get_time_series_for_geometry


class TsTest(unittest.TestCase):

    def setUp(self):
        # max = 25 * 180 * 360
        shape = 25, 180, 360
        dims = 'time', 'lat', 'lon'
        self._cube = new_cube(time_periods=25,
                              variables=dict(
                                  A=xr.DataArray(np.broadcast_to(np.linspace(1, 25, 25).reshape(25, 1, 1), shape),
                                                 dims=dims),
                                  B=xr.DataArray(np.broadcast_to(np.linspace(0, 1, 25).reshape(25, 1, 1), shape),
                                                 dims=dims)
                              ))
        self._cube = self._cube.chunk(chunks=dict(time=1, lat=180, lon=180))

    def test_point(self):
        ts_ds = get_time_series_for_point(self._cube, dict(type='Point', coordinates=[20.0, 10.0]))

        self.assertIsNotNone(ts_ds)
        self.assertIn('A', ts_ds)
        self.assertIn('B', ts_ds)
        self.assertEqual(('time',), ts_ds.A.dims)
        self.assertEqual(('time',), ts_ds.B.dims)
        self.assertEqual(25, ts_ds.A.size)
        self.assertEqual(25, ts_ds.B.size)
        self.assertEqual(1, ts_ds.attrs.get('max_number_of_observations'))
        np.testing.assert_almost_equal(np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.,
                                                 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.]),
                                       ts_ds.A.values)

    def test_polygon(self):
        ts_ds = get_time_series_for_geometry(self._cube,
                                             dict(type='Polygon',
                                                  coordinates=[[[20.0, 10.0],
                                                                [20.0, 20.0],
                                                                [10.0, 20.0],
                                                                [10.0, 10.0],
                                                                [20.0, 10.0]]]),
                                             include_count=True,
                                             include_stdev=True,
                                             use_groupby=False)
        self.assertIsNotNone(ts_ds)
        self.assertIn('A', ts_ds)
        self.assertIn('B', ts_ds)
        self.assertIn('A_stdev', ts_ds)
        self.assertIn('B_stdev', ts_ds)
        self.assertIn('A_count', ts_ds)
        self.assertIn('B_count', ts_ds)
        self.assertEqual(('time',), ts_ds.A.dims)
        self.assertEqual(('time',), ts_ds.B.dims)
        self.assertEqual(25, ts_ds.A.size)
        self.assertEqual(25, ts_ds.B.size)
        self.assertEqual(110, ts_ds.attrs.get('max_number_of_observations'))
        np.testing.assert_almost_equal(np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.,
                                                 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.]),
                                       ts_ds.A.values)

    def test_polygon_using_groupby(self):
        ts_ds = get_time_series_for_geometry(self._cube,
                                             dict(type='Polygon',
                                                  coordinates=[[[20.0, 10.0],
                                                                [20.0, 20.0],
                                                                [10.0, 20.0],
                                                                [10.0, 10.0],
                                                                [20.0, 10.0]]]),
                                             include_count=True,
                                             include_stdev=True,
                                             use_groupby=True)
        self.assertIsNotNone(ts_ds)
        self.assertIn('A', ts_ds)
        self.assertIn('B', ts_ds)
        self.assertIn('A_stdev', ts_ds)
        self.assertIn('B_stdev', ts_ds)
        self.assertIn('A_count', ts_ds)
        self.assertIn('B_count', ts_ds)
        self.assertEqual(('time',), ts_ds.A.dims)
        self.assertEqual(('time',), ts_ds.B.dims)
        self.assertEqual(25, ts_ds.A.size)
        self.assertEqual(25, ts_ds.B.size)
        self.assertEqual(110, ts_ds.attrs.get('max_number_of_observations'))
        np.testing.assert_almost_equal(np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.,
                                                 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.]),
                                       ts_ds.A.values)
