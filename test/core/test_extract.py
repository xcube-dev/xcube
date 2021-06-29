import unittest

import numpy as np
import pandas as pd
import xarray as xr

from xcube.core.extract import get_cube_point_indexes
from xcube.core.extract import get_cube_values_for_points
from xcube.core.extract import get_dataset_indexes
from xcube.core.new import new_cube


# noinspection PyMethodMayBeStatic
class GetCubeValuesForPointsTest(unittest.TestCase):
    def _new_test_cube(self):
        return new_cube(width=2000,
                        height=1000,
                        x_start=0.0,
                        y_start=50.0,
                        x_res=4.0 / 2000,
                        time_start="2010-01-01",
                        time_periods=20,
                        variables=dict(precipitation=0.6, temperature=276.2))

    def _new_test_points(self):
        return dict(time=np.array(["2010-01-04", "2009-04-09",
                                   "2010-01-08", "2010-01-06",
                                   "2010-01-20", "2010-01-01",
                                   "2010-01-05", "2010-01-03",
                                   ], dtype="datetime64[ns]"),
                    # Note "lat" is a list (see issue #434)
                    lat=[50.0, 51.3, 49.7, 50.1, 51.9, 50.8, 50.2, 52.0],
                    lon=np.array([0.0, 0.1, 0.4, 2.9, 1.6, 0.7, -0.5, 4.0]))

    def test_get_cube_point_values_from_data_frame(self):
        self._test_get_cube_point_values(pd.DataFrame(self._new_test_points()))

    def test_get_cube_point_values_from_dataset(self):
        self._test_get_cube_point_values(xr.Dataset(self._new_test_points()))

    def test_get_cube_point_values_from_dict(self):
        self._test_get_cube_point_values(self._new_test_points())

    def _test_get_cube_point_values(self, points):
        cube = self._new_test_cube()

        expected_prec_values = [0.6, np.nan, np.nan,
                                0.6, 0.6, 0.6, np.nan, 0.6]
        expected_temp_values = [276.2, np.nan, np.nan,
                                276.2, 276.2, 276.2, np.nan, 276.2]
        expected_time_index = [3, -1, 7, 5, 19, 0, 4, 2]
        expected_lat_index = [0, 650, -1, 50, 950, 400, 100, 999]
        expected_lon_index = [0, 50, 200, 1450, 800, 349, -1, 1999]
        values = get_cube_values_for_points(cube, points, include_indexes=True)
        self.assertIsInstance(values, xr.Dataset)
        self.assertEqual(5, len(values.data_vars))
        self.assertEqual({
            'precipitation',
            'temperature',
            'lat_index',
            'lon_index',
            'time_index'},
            set(values.data_vars))
        np.testing.assert_array_almost_equal(np.array(expected_prec_values,
                                                      dtype=np.float64),
                                             values["precipitation"].values)
        np.testing.assert_array_almost_equal(np.array(expected_temp_values,
                                                      dtype=np.float64),
                                             values["temperature"].values)
        np.testing.assert_array_equal(np.array(expected_time_index,
                                               dtype=np.int64),
                                      values["time_index"].values)
        np.testing.assert_array_equal(np.array(expected_lat_index,
                                               dtype=np.int64),
                                      values["lat_index"].values)
        np.testing.assert_array_equal(np.array(expected_lon_index,
                                               dtype=np.int64),
                                      values["lon_index"].values)

    def test_get_cube_point_indexes(self):
        cube = self._new_test_cube()

        expected_time_index = [3., np.nan, 7.,
                               5., 19., 0., 4., 2.]
        expected_lat_index = [0., 650., np.nan,
                              50., 950, 400., 100., 1000. - 1e-9]
        expected_lon_index = [0., 50., 200.,
                              1450., 800., 350., np.nan, 2000. - 1e-9]

        indexes = get_cube_point_indexes(cube, self._new_test_points())
        self.assertIsInstance(indexes, xr.Dataset)
        self.assertEqual(3, len(indexes.data_vars))
        self.assertEqual({'time_index', 'lat_index', 'lon_index'},
                         set(indexes.data_vars))
        np.testing.assert_array_almost_equal(np.array(expected_time_index,
                                                      dtype=np.float64),
                                             indexes["time_index"].values)
        np.testing.assert_array_almost_equal(np.array(expected_lat_index,
                                                      dtype=np.float64),
                                             indexes["lat_index"].values)
        np.testing.assert_array_almost_equal(np.array(expected_lon_index,
                                                      dtype=np.float64),
                                             indexes["lon_index"].values)


# noinspection PyMethodMayBeStatic
class GetDatasetIndexesTest(unittest.TestCase):

    def test_get_dataset_indexes_for_single_cell(self):
        dataset = new_cube(width=360, height=180, drop_bounds=True)
        cell = dataset.isel(time=2, lat=20, lon=30)
        with self.assertRaises(ValueError) as cm:
            get_dataset_indexes(cell, "lon", np.array([-149.5]))
        self.assertEqual("cannot determine cell boundaries"
                         " for coordinate variable 'lon' of size 1",
                         f"{cm.exception}")

    def test_get_dataset_indexes_with_bounds(self):
        dataset = new_cube(width=360, height=180,
                           drop_bounds=False)
        self._assert_get_dataset_indexes_works(dataset)

    def test_get_dataset_indexes_without_bounds(self):
        dataset = new_cube(width=360, height=180,
                           drop_bounds=True)
        self._assert_get_dataset_indexes_works(dataset)

    def test_get_dataset_indexes_with_bounds_inverse_lat(self):
        dataset = new_cube(width=360, height=180,
                           inverse_y=True, drop_bounds=False)
        self._assert_get_dataset_indexes_works(dataset, inverse_lat=True)

    def test_get_dataset_indexes_without_bounds_inverse_lat(self):
        dataset = new_cube(width=360, height=180,
                           inverse_y=True, drop_bounds=True)
        self._assert_get_dataset_indexes_works(dataset, inverse_lat=True)

    def _assert_get_dataset_indexes_works(self, dataset, inverse_lat=False):
        # - lon -------------------------------------

        lon_coords = np.array([-190, -180., -179,
                               -10.4, 0., 10.4, 179., 180.0, 190])
        expected_lon_int64 = np.array([-1, 0, 1,
                                       169, 180, 190,
                                       359, 359, -1],
                                      dtype=np.int64)
        expected_lon_float64 = np.array([np.nan, 0., 1.,
                                         169.6, 180., 190.4,
                                         359., 360., np.nan],
                                        dtype=np.float64)

        indexes = get_dataset_indexes(dataset, "lon", lon_coords,
                                      index_dtype=np.int32)
        np.testing.assert_array_equal(indexes, expected_lon_int64)
        indexes = get_dataset_indexes(dataset, "lon", lon_coords)
        np.testing.assert_array_almost_equal(indexes, expected_lon_float64)

        # - lat -------------------------------------

        lat_coords = np.array([-100, -90., -89,
                               -10.4, 0., 10.4, 89., 90.0, 100])
        if not inverse_lat:
            expected_lat_int64 = np.array([-1, 0, 1,
                                           79, 90, 100, 179, 179, -1],
                                          dtype=np.int64)
            expected_lat_float64 = np.array([np.nan, 0., 1., 79.6,
                                             90., 100.4, 179.,
                                             180., np.nan],
                                            dtype=np.float64)
        else:
            expected_lat_int64 = np.array([-1, 179, 179,
                                           100, 90, 79, 1, 0, -1],
                                          dtype=np.int64)
            expected_lat_float64 = np.array([np.nan, 180., 179.,
                                             100.4, 90., 79.6, 1., 0., np.nan],
                                            dtype=np.float64)

        indexes = get_dataset_indexes(dataset, "lat", lat_coords,
                                      index_dtype=np.int32)
        np.testing.assert_array_equal(indexes, expected_lat_int64)
        indexes = get_dataset_indexes(dataset, "lat", lat_coords)
        np.testing.assert_array_almost_equal(indexes, expected_lat_float64)

        # - time ------------------------------------

        time_coords = np.array(
            ["2010-01-03", "2010-01-02T23:15", "2010-05-15",
             "2010-01-01T12:00", "2010-01-05", "2009-12-07"],
            dtype="datetime64[ns]")
        expected_time_int64 = np.array([2, 1, -1, 0, 4, -1],
                                       dtype=np.int64)
        expected_time_float64 = np.array([2., 1.96875, np.nan,
                                          0.5, 4., np.nan],
                                         dtype=np.float64)

        indexes = get_dataset_indexes(dataset, "time", time_coords,
                                      index_dtype=np.int32)
        np.testing.assert_array_equal(indexes, expected_time_int64)
        indexes = get_dataset_indexes(dataset, "time", time_coords)
        np.testing.assert_array_almost_equal(indexes, expected_time_float64)
