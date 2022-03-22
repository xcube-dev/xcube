import unittest
from typing import Set

import numpy as np
import pyproj
import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.core.new import new_cube
from xcube.core.timeseries import get_time_series

POINT_GEOMETRY = dict(type='Point', coordinates=[20.0, 10.0])
POLYGON_GEOMETRY = dict(type='Polygon', coordinates=[[[20.0, 10.0],
                                                      [20.0, 20.0],
                                                      [10.0, 20.0],
                                                      [10.0, 10.0],
                                                      [20.0, 10.0]]])


class GetTimeSeriesTest(unittest.TestCase):

    def test_point(self):
        ts_ds = get_time_series(self.cube, geometry=POINT_GEOMETRY)
        self.assert_dataset_ok(ts_ds, 1, {'A', 'B'})

    def test_polygon(self):
        ts_ds = get_time_series(self.cube, geometry=POLYGON_GEOMETRY)
        self.assert_dataset_ok(ts_ds, 100, {'A_mean', 'B_mean'})

    def test_polygon_with_grid_mapping(self):
        crs1 = pyproj.CRS.from_string('EPSG:4326')  # = Geographic
        crs2 = pyproj.CRS.from_string('+proj=mill +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +R_A +datum=WGS84 +units=m +no_defs')  # = World Miller
        t = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
        cube = self.cube.rename(dict(lat='y', lon='x'))
        x, _ = t.transform(
            cube.x,
            np.zeros(cube.x.size)
        )
        _, y = t.transform(
            np.zeros(cube.y.size),
            cube.y,
        )
        cube = cube.assign_coords(
            x=xr.DataArray(x, dims='x'),
            y=xr.DataArray(y, dims='y'),
        )
        cube = cube.assign(crs=xr.DataArray(0, attrs=crs2.to_cf()))
        grid_mapping = GridMapping.from_dataset(cube)
        ts_ds = get_time_series(cube,
                                grid_mapping=grid_mapping,
                                geometry=POLYGON_GEOMETRY)
        self.assertIsInstance(ts_ds, xr.Dataset)
        # TODO (forman): we must actually get a valid
        #  result here, but we don't
        # self.assert_dataset_ok(ts_ds, 100, {'A_mean', 'B_mean'})

    def test_polygon_deprecated_cube_asserted(self):
        ts_ds = get_time_series(self.cube, geometry=POLYGON_GEOMETRY,
                                cube_asserted=True)
        self.assert_dataset_ok(ts_ds, 100, {'A_mean', 'B_mean'})

    def test_polygon_agg_median_mean_std(self):
        ts_ds = get_time_series(self.cube,
                                geometry=POLYGON_GEOMETRY,
                                agg_methods=['mean', 'std', 'median'])
        self.assert_dataset_ok(
            ts_ds, 100,
            {'A_mean', 'A_median', 'A_std', 'B_mean', 'B_median', 'B_std'}
        )

    def test_polygon_agg_median_mean_std_groupby(self):
        ts_ds = get_time_series(self.cube,
                                geometry=POLYGON_GEOMETRY,
                                agg_methods=['mean', 'std', 'median'],
                                use_groupby=True)
        self.assert_dataset_ok(
            ts_ds, 100,
            {'A_mean', 'A_median', 'A_std', 'B_mean', 'B_median', 'B_std'}
        )

    def test_polygon_agg_mean_count(self):
        ts_ds = get_time_series(self.cube,
                                geometry=POLYGON_GEOMETRY,
                                agg_methods=['mean', 'count'])
        self.assert_dataset_ok(
            ts_ds, 100,
            {'A_mean', 'A_count', 'B_mean', 'B_count'}
        )

    def test_polygon_agg_mean_std_var_subs(self):
        ts_ds = get_time_series(self.cube,
                                geometry=POLYGON_GEOMETRY,
                                var_names=['B'],
                                agg_methods=['mean', 'std'])
        self.assert_dataset_ok(ts_ds, 100, {'B_mean', 'B_std'})

    def test_no_vars(self):
        ts_ds = get_time_series(self.cube,
                                geometry=POLYGON_GEOMETRY,
                                var_names=[])
        self.assertIsNone(ts_ds)

    def test_illegal_agg_methods(self):
        with self.assertRaises(ValueError) as cm:
            get_time_series(self.cube,
                            geometry=POLYGON_GEOMETRY,
                            agg_methods=['mean', 'median', 'stdev'])
        self.assertEqual('invalid aggregation method: stdev',
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            get_time_series(self.cube,
                            geometry=POLYGON_GEOMETRY,
                            agg_methods=['median', 'stdev', 'avg'])
        self.assertEqual('invalid aggregation methods: avg, stdev',
                         f'{cm.exception}')

    def setUp(self):
        shape = 25, 180, 360
        dims = 'time', 'lat', 'lon'
        self.ts_a = np.linspace(1, 25, 25)
        self.ts_a_mean = np.linspace(1, 25, 25)
        self.ts_a_count = np.array(25 * [100])
        self.ts_a_std = np.array(25 * [0.0])
        self.ts_b = np.linspace(0, 1, 25)
        self.ts_b_mean = np.linspace(0, 1, 25)
        self.ts_b_count = np.array(25 * [100])
        self.ts_b_std = np.array(25 * [0.0])
        self.cube = new_cube(
            time_periods=25,
            variables=dict(
                A=xr.DataArray(
                    np.broadcast_to(self.ts_a.reshape((25, 1, 1)), shape),
                    dims=dims),
                B=xr.DataArray(
                    np.broadcast_to(self.ts_b.reshape((25, 1, 1)), shape),
                    dims=dims)
            )
        )
        self.cube = self.cube.chunk(chunks=dict(time=1, lat=180, lon=180))

    def assert_dataset_ok(self, ts_ds: xr.Dataset,
                          expected_max_number_of_observations: int = 0,
                          expected_var_names: Set = None):
        expected_var_names = expected_var_names or set()
        self.assertIsNotNone(ts_ds)
        self.assertEqual(expected_max_number_of_observations, ts_ds.attrs.get('max_number_of_observations'))
        self.assert_variable_ok(ts_ds, 'A', expected_var_names, self.ts_a)
        self.assert_variable_ok(ts_ds, 'A_mean', expected_var_names, self.ts_a_mean)
        self.assert_variable_ok(ts_ds, 'A_count', expected_var_names, self.ts_a_count)
        self.assert_variable_ok(ts_ds, 'A_std', expected_var_names, self.ts_a_std)
        self.assert_variable_ok(ts_ds, 'B', expected_var_names, self.ts_b)
        self.assert_variable_ok(ts_ds, 'B_mean', expected_var_names, self.ts_b_mean)
        self.assert_variable_ok(ts_ds, 'B_count', expected_var_names, self.ts_b_count)
        self.assert_variable_ok(ts_ds, 'B_std', expected_var_names, self.ts_b_std)

    def assert_variable_ok(self, ts_ds, name, expected_var_names: Set, expected_values):
        if name in expected_var_names:
            self.assertIn(name, ts_ds)
            self.assertEqual(('time',), ts_ds[name].dims)
            self.assertEqual(25, ts_ds[name].size)
            np.testing.assert_almost_equal(ts_ds[name].values, expected_values)
        else:
            self.assertNotIn(name, ts_ds)
