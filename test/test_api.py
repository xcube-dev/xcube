import os
import unittest

import numpy as np
import xarray as xr

from test.sampledata import new_test_dataset
from xcube.api import open_dataset, read_dataset, write_dataset, dump_dataset, chunk_dataset, verify_cube, \
    assert_cube, get_cube_point_indexes, get_cube_values_for_points, new_cube, get_dataset_indexes, \
    _get_block_interp_arrays

TEST_NC_FILE = "test.nc"


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


TEST_NC_FILE_2 = "test-2.nc"


class OpenReadWriteDatasetTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.dataset = new_cube(variables=dict(precipitation=0.2, temperature=279.1))
        self.dataset.to_netcdf(TEST_NC_FILE, mode="w")
        self.dataset.close()

    def tearDown(self):
        self.dataset = None
        os.remove(TEST_NC_FILE)
        super().tearDown()

    def test_open_dataset(self):
        with open_dataset(TEST_NC_FILE) as ds:
            self.assertIsNotNone(ds)
            np.testing.assert_array_equal(ds.time.values, self.dataset.time.values)
            np.testing.assert_array_equal(ds.lat.values, self.dataset.lat.values)
            np.testing.assert_array_equal(ds.lon.values, self.dataset.lon.values)

    def test_read_dataset(self):
        ds = read_dataset(TEST_NC_FILE)
        self.assertIsNotNone(ds)
        np.testing.assert_array_equal(ds.time.values, self.dataset.time.values)
        np.testing.assert_array_equal(ds.lat.values, self.dataset.lat.values)
        np.testing.assert_array_equal(ds.lon.values, self.dataset.lon.values)
        ds.close()

    def test_write_dataset(self):

        dataset = new_cube()
        try:
            write_dataset(dataset, TEST_NC_FILE_2)
            self.assertTrue(os.path.isfile(TEST_NC_FILE_2))
        finally:
            if os.path.isfile(TEST_NC_FILE_2):
                os.remove(TEST_NC_FILE_2)


class ChunkDatasetTest(unittest.TestCase):
    def test_chunk_dataset(self):
        dataset = new_test_dataset(["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-04", "2010-01-05"],
                                   precipitation=0.4, temperature=275.2)

        chunked_dataset = chunk_dataset(dataset,
                                        chunk_sizes=dict(time=1, lat=10, lon=20),
                                        format_name="zarr")
        self.assertEqual({'chunks': (1, 10, 20)}, chunked_dataset.precipitation.encoding)
        self.assertEqual({'chunks': (1, 10, 20)}, chunked_dataset.temperature.encoding)

        chunked_dataset = chunk_dataset(dataset,
                                        chunk_sizes=dict(time=1, lat=20, lon=40),
                                        format_name="netcdf4")
        self.assertEqual({'chunksizes': (1, 20, 40)}, chunked_dataset.precipitation.encoding)
        self.assertEqual({'chunksizes': (1, 20, 40)}, chunked_dataset.temperature.encoding)

        chunked_dataset = chunk_dataset(dataset,
                                        chunk_sizes=dict(time=1, lat=20, lon=40))
        self.assertEqual({}, chunked_dataset.precipitation.encoding)
        self.assertEqual({}, chunked_dataset.temperature.encoding)

    def test_unchunk_dataset(self):
        dataset = new_test_dataset(["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-04", "2010-01-05"],
                                   precipitation=0.4, temperature=275.2)

        for var in dataset.data_vars.values():
            var.encoding.update({"chunks": (5, 180, 360), "_FillValue": -999.0})

        chunked_dataset = chunk_dataset(dataset, format_name="zarr")
        self.assertEqual({"_FillValue": -999.0}, chunked_dataset.precipitation.encoding)
        self.assertEqual({"_FillValue": -999.0}, chunked_dataset.temperature.encoding)


class DumpDatasetTest(unittest.TestCase):
    def test_dump_dataset(self):
        dataset = new_test_dataset(["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-04", "2010-01-05"],
                                   precipitation=0.4, temperature=275.2)
        for var in dataset.variables.values():
            var.encoding.update({"_FillValue": 999.0})

        print(dataset.dims)

        text = dump_dataset(dataset)
        self.assertIn("<xarray.Dataset>", text)
        self.assertIn("Dimensions:        (lat: 180, lon: 360, time: 5)\n", text)
        self.assertIn("Coordinates:\n", text)
        self.assertIn("  * lon            (lon) float64 ", text)
        self.assertIn("Data variables:\n", text)
        self.assertIn("    precipitation  (time, lat, lon) float64 ", text)
        self.assertNotIn("Encoding for coordinate variable 'lat':\n", text)
        self.assertNotIn("Encoding for data variable 'temperature':\n", text)
        self.assertNotIn("    _FillValue:  999.0\n", text)

        text = dump_dataset(dataset, show_var_encoding=True)
        self.assertIn("<xarray.Dataset>", text)
        self.assertIn("Dimensions:        (lat: 180, lon: 360, time: 5)\n", text)
        self.assertIn("Coordinates:\n", text)
        self.assertIn("  * lon            (lon) float64 ", text)
        self.assertIn("Data variables:\n", text)
        self.assertIn("    precipitation  (time, lat, lon) float64 ", text)
        self.assertIn("Encoding for coordinate variable 'lat':\n", text)
        self.assertIn("Encoding for data variable 'temperature':\n", text)
        self.assertIn("    _FillValue:  999.0\n", text)

        text = dump_dataset(dataset, ["precipitation"])
        self.assertIn("<xarray.DataArray 'precipitation' (time: 5, lat: 180, lon: 360)>\n", text)
        self.assertNotIn("Encoding:\n", text)
        self.assertNotIn("    _FillValue:  999.0", text)

        text = dump_dataset(dataset, ["precipitation"], show_var_encoding=True)
        self.assertIn("<xarray.DataArray 'precipitation' (time: 5, lat: 180, lon: 360)>\n", text)
        self.assertIn("Encoding:\n", text)
        self.assertIn("    _FillValue:  999.0", text)


class AssertAndVerifyCubeTest(unittest.TestCase):

    def test_assert_cube(self):
        cube = new_cube(variables=dict(precipitation=0.5))
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


# noinspection PyMethodMayBeStatic
class ExtractPointsTest(unittest.TestCase):
    def _new_test_cube(self):
        return new_cube(width=2000,
                        height=1000,
                        lon_start=0.0,
                        lat_start=50.0,
                        spatial_res=4.0 / 2000,
                        time_start="2010-01-01",
                        time_periods=20,
                        variables=dict(precipitation=0.6, temperature=276.2))

    def test_get_cube_point_values(self):
        cube = self._new_test_cube()
        values = get_cube_values_for_points(cube,
                                            dict(time=np.array(["2010-01-04", "2010-01-02",
                                                                "2010-01-08", "2010-01-02",
                                                                "2010-01-02", "2010-01-01",
                                                                "2010-01-05", "2010-01-03",
                                                                ], dtype="datetime64[ns]"),
                                                 lat=np.array([50.0, 51.3, 49.7, 50.1, 51.9, 50.8, 50.2, 52.0]),
                                                 lon=np.array([0.0, 0.1, 0.4, 2.9, 1.6, 0.7, -0.5, 4.0]),
                                                 ))
        print(values)

    def test_get_cube_point_indexes(self):
        cube = self._new_test_cube()
        indexes = get_cube_point_indexes(cube,
                                         dict(time=np.array(["2010-01-04", "2009-11-23",
                                                             "2010-01-08", "2010-01-06",
                                                             "2010-01-20", "2010-01-01",
                                                             "2010-01-05", "2010-01-03",
                                                             ], dtype="datetime64[ns]"),
                                              lat=np.array([50.0, 51.3, 49.7, 50.1, 51.9, 50.8, 50.2, 52.0]),
                                              lon=np.array([0.0, 0.1, 0.4, 2.9, 1.6, 0.7, -0.5, 4.0]),
                                              ),
                                         include_fractions=True)

        self.assertEqual(8, len(indexes))
        self.assertEqual(6, len(indexes.columns))
        self.assertEqual(['time_index', 'time_fraction',
                          'lat_index', 'lat_fraction',
                          'lon_index', 'lon_fraction'],
                         [c for c in indexes])
        np.testing.assert_array_equal(np.array([3, -1, 7, 5, 19, 0, 4, 2], dtype=np.int64),
                                      indexes["time_index"].values)
        np.testing.assert_array_equal(np.array([0., np.nan, 0., 0., 0., 0., 0., 0.], dtype=np.float64),
                                      indexes["time_fraction"].values)
        np.testing.assert_array_equal(np.array([0, 650, -1, 50, 950, 400, 100, 999], dtype=np.int64),
                                      indexes["lat_index"].values)
        np.testing.assert_array_equal(np.array([0., 0., np.nan, 0., 0., 0., 0., 1.], dtype=np.float64),
                                      indexes["lat_fraction"].values)
        np.testing.assert_array_equal(np.array([0, 50, 200, 1450, 800, 349, -1, 1999], dtype=np.int64),
                                      indexes["lon_index"].values)
        np.testing.assert_array_equal(np.array([0., 0., 0., 0., 0., 1., np.nan, 1.], dtype=np.float64),
                                      indexes["lon_fraction"].values)


# noinspection PyMethodMayBeStatic
class CoordIndexTest(unittest.TestCase):

    def test_get_coord_index_for_single_cell(self):
        dataset = new_cube(width=360, height=180, drop_bounds=True)
        cell = dataset.isel(time=2, lat=20, lon=30)
        with self.assertRaises(ValueError) as cm:
            get_dataset_indexes(cell, "lon", np.array([-149.5]))
        self.assertEqual("cannot determine cell boundaries for coordinate variable 'lon' of size 1",
                         f"{cm.exception}")

    def test_get_coord_index_with_bounds(self):
        dataset = new_cube(width=360, height=180, drop_bounds=False)
        self._assert_get_coord_index(dataset)

    def test_get_coord_index_without_bounds(self):
        dataset = new_cube(width=360, height=180, drop_bounds=True)
        self._assert_get_coord_index(dataset)

    def test_get_coord_index_with_bounds_reverse_lat(self):
        dataset = new_cube(width=360, height=180, drop_bounds=False)
        lat = dataset.lat.values[::-1]
        lat_bnds = dataset.lat_bnds.values[::-1, ::-1]
        dataset.coords["lat"] = xr.DataArray(lat, dims=("lat",))
        dataset.coords["lat_bnds"] = xr.DataArray(lat_bnds, dims=("lat", "bnds"))
        self._assert_get_coord_index(dataset, reverse_lat=True)

    def test_get_coord_index_without_bounds_reverse_lat(self):
        dataset = new_cube(width=360, height=180, drop_bounds=True)
        lat = dataset.lat.values[::-1]
        dataset.coords["lat"] = xr.DataArray(lat, dims=("lat",))
        self._assert_get_coord_index(dataset, reverse_lat=True)

    def _assert_get_coord_index(self, dataset, reverse_lat=False):
        lon_coords = np.array([-190, -180., -179, -10.4, 0., 10.4, 179., 180.0, 190])
        expected_lon_indexes = np.array([-1, 0, 1, 169, 180, 190, 359, 359, -1], dtype=np.int64)
        expected_lon_fractions = np.array([np.nan, 0., 0., 0.6, 0., 0.4, 0., 1., np.nan], dtype=np.float64)

        lat_coords = np.array([-100, -90., -89, -10.4, 0., 10.4, 89., 90.0, 100])
        expected_lat_indexes = np.array([-1, 0, 1, 79, 90, 100, 179, 179, -1], dtype=np.int64)
        expected_lat_fractions = np.array([np.nan, 0., 0., 0.6, 0., 0.4, 0., 1., np.nan], dtype=np.float64)
        if reverse_lat:
            expected_lat_indexes = expected_lat_indexes[::-1]
            expected_lat_fractions = expected_lat_fractions[::-1]

        time_coords = np.array(
            ["2010-01-03", "2010-01-02T23:15", "2010-05-15", "2010-01-01T12:00", "2010-01-05", "2009-12-07"],
            dtype="datetime64[ns]")
        expected_time_indexes = np.array([2, 1, -1, 0, 4, -1], dtype=np.int64)
        expected_time_fractions = np.array([0., 0.96875, np.nan, 0.5, 0., np.nan], dtype=np.float64)

        indexes, fractions = get_dataset_indexes(dataset, "lon", lon_coords)
        np.testing.assert_array_equal(indexes, expected_lon_indexes)
        np.testing.assert_array_almost_equal(fractions, expected_lon_fractions)

        indexes, fractions = get_dataset_indexes(dataset, "lat", lat_coords)
        np.testing.assert_array_equal(indexes, expected_lat_indexes)
        np.testing.assert_array_almost_equal(fractions, expected_lat_fractions)

        indexes, fractions = get_dataset_indexes(dataset, "time", time_coords)
        np.testing.assert_array_equal(indexes, expected_time_indexes)
        np.testing.assert_array_almost_equal(fractions, expected_time_fractions)


class ChunksTest(unittest.TestCase):

    def test_get_block_interp_arrays(self):
        chunks = ((1, 1, 1), (375, 375, 375, 375), (509, 509, 509, 509))
        coords_indexes = _get_block_interp_arrays(chunks)
        coords, indexes = zip(*coords_indexes)

        np.testing.assert_array_equal(coords[0], np.array([0, 1, 2, 3]))
        np.testing.assert_array_equal(coords[1], np.array([0, 375, 750, 1125, 1500]))
        np.testing.assert_array_equal(coords[2], np.array([0, 509, 1018, 1527, 2036]))

        np.testing.assert_array_equal(indexes[0], np.array([0, 1, 2, 3]))
        np.testing.assert_array_equal(indexes[1], np.array([0, 1, 2, 3, 4]))
        np.testing.assert_array_equal(indexes[2], np.array([0, 1, 2, 3, 4]))

        result = np.interp([-1, 0, 1, 2, 3, 4],
                           coords[0], indexes[0], left=-1, right=-1).astype(dtype=np.int64)
        np.testing.assert_array_equal(result, np.array([-1, 0, 1, 2, 3, -1]))

        result = np.interp([-1, 0, 1, 374, 375, 376, 749, 750, 751, 1124, 1125, 1126, 1499, 1500, 1501],
                           coords[1], indexes[1], left=-1, right=-1).astype(dtype=np.int64)
        np.testing.assert_array_equal(result, np.array([-1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, -1]))

        result = np.interp([0, 509, 1018, 1527, 2036],
                           coords[2], indexes[2], left=-1, right=-1).astype(dtype=np.int64)
        np.testing.assert_array_equal(result, np.array([0, 1, 2, 3, 4]))

        block_indexes = tuple(tuple((c[i], c[i + 1]) for i in range(len(c) - 1)) for c in coords)
        print(block_indexes)

        import itertools
        block_dim_slices = tuple(itertools.product(*block_indexes))
        for dim_slices in block_dim_slices:
            print(dim_slices)

        block_dim_slices = np.array(block_dim_slices, dtype=np.int32)
        for dim_slices in block_dim_slices:
            print(dim_slices[:, 0], dim_slices[:, 1])

    def test_apply_ufunc(self):

        cube = new_cube(variables=dict(a=6.1, b=8.2))
        cube = cube.chunk(chunks=90)

        stuff = []

        def my_func(x: np.ndarray, **kwargs):
            stuff.append((type(x), x.shape, x.strides, x.strides, kwargs))
            return x

        cube2 = xr.apply_ufunc(my_func,
                               cube,
                               dask="parallelized",
                               output_dtypes=[np.float64],
                               kwargs=dict(urgs=8.1))
        # print(cube2.a.load())
        # print(stuff)
