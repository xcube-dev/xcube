import unittest

import numpy as np
import xarray as xr
import zarr

from test.core.diagnosticstore import DiagnosticStore, logging_observer
from xcube.core.chunk import chunk_dataset
from xcube.core.dsio import rimraf
from xcube.core.new import new_cube
from xcube.core.timeslice import find_time_slice, append_time_slice, insert_time_slice, replace_time_slice


class TimeSliceTest(unittest.TestCase):
    CUBE_PATH = 'test-cube.zarr'
    CUBE_PATH_2 = 'test-cube-2.zarr'

    def setUp(self) -> None:
        rimraf(self.CUBE_PATH)
        rimraf(self.CUBE_PATH_2)

    def tearDown(self) -> None:
        rimraf(self.CUBE_PATH)
        rimraf(self.CUBE_PATH_2)

    def write_slice(self, start_date):
        self.write_cube(start_date, 1)

    def make_slice(self, start_date) -> xr.Dataset:
        return self.make_cube(start_date, 1)

    def write_cube(self, start_date, num_days: int):
        cube = self.make_cube(start_date, num_days)
        cube.to_zarr(self.CUBE_PATH)

    def make_cube(self, start_date, num_days: int) -> xr.Dataset:
        cube = new_cube(time_periods=num_days,
                        time_freq='1D',
                        time_start=start_date,
                        variables=dict(precipitation=0.1,
                                       temperature=270.5,
                                       soil_moisture=0.2))
        chunk_sizes = dict(time=1, lat=90, lon=90)
        cube = chunk_dataset(cube, chunk_sizes, format_name='zarr')
        return cube

    def test_find_time_slice(self):
        self.write_cube('2019-01-01', 10)

        # Cube does not exists --> write new
        result = find_time_slice(self.CUBE_PATH_2, np.datetime64('2018-12-30T13:00'))
        self.assertEqual((-1, 'create'), result)

        # Before first step --> insert before 0
        result = find_time_slice(self.CUBE_PATH, np.datetime64('2018-12-30T13:00'))
        self.assertEqual((0, 'insert'), result)

        # After first step --> insert before 1
        result = find_time_slice(self.CUBE_PATH, np.datetime64('2019-01-02'))
        self.assertEqual((1, 'insert'), result)

        # In-between --> insert before 5
        result = find_time_slice(self.CUBE_PATH, np.datetime64('2019-01-06T10:00:00'))
        self.assertEqual((5, 'insert'), result)

        # In-between at existing time-stamp --> replace 5
        result = find_time_slice(self.CUBE_PATH, np.datetime64('2019-01-06T12:00:00'))
        self.assertEqual((5, 'replace'), result)

        # After last step --> append
        result = find_time_slice(self.CUBE_PATH, np.datetime64('2019-01-12'))
        self.assertEqual((-1, 'append'), result)

    def test_append_time_slice(self):
        self.write_slice('2019-01-01T14:30')

        append_time_slice(self.CUBE_PATH, self.make_slice('2019-01-02T01:14'))
        append_time_slice(self.CUBE_PATH, self.make_slice('2019-01-02T05:16'))
        append_time_slice(self.CUBE_PATH, self.make_slice('2019-01-03T03:50'))

        cube = xr.open_zarr(self.CUBE_PATH)
        expected = np.array(['2019-01-02T02:30',
                             '2019-01-02T13:14',
                             '2019-01-02T17:16',
                             '2019-01-03T15:50'], dtype=cube.time.dtype)
        self.assertEqual(4, cube.time.size)
        self.assertEqual(None, cube.time.chunks)
        np.testing.assert_equal(cube.time.values, expected)

    def test_insert_time_slice(self):
        self.write_cube('2019-01-02T00:00:00', 10)

        cube = xr.open_zarr(self.CUBE_PATH)
        expected = np.array(
            ['2019-01-02T12:00:00', '2019-01-03T12:00:00',
             '2019-01-04T12:00:00', '2019-01-05T12:00:00',
             '2019-01-06T12:00:00', '2019-01-07T12:00:00',
             '2019-01-08T12:00:00', '2019-01-09T12:00:00',
             '2019-01-10T12:00:00', '2019-01-11T12:00:00'],
            dtype=cube.time.dtype)
        np.testing.assert_equal(cube.time.values, expected)

        insert_time_slice(self.CUBE_PATH, 5,
                          self.make_slice('2019-01-06T02:00:00'))

        cube = xr.open_zarr(self.CUBE_PATH)
        expected = np.array(
            ['2019-01-02T12:00:00', '2019-01-03T12:00:00',
             '2019-01-04T12:00:00', '2019-01-05T12:00:00',
             '2019-01-06T12:00:00', '2019-01-06T14:00:00', '2019-01-07T12:00:00',
             '2019-01-08T12:00:00', '2019-01-09T12:00:00',
             '2019-01-10T12:00:00', '2019-01-11T12:00:00'],
            dtype=cube.time.dtype)
        np.testing.assert_equal(cube.time.values, expected)

        insert_time_slice(self.CUBE_PATH, 10, self.make_slice('2019-01-10T02:00'))
        insert_time_slice(self.CUBE_PATH, 0, self.make_slice('2019-01-01T02:00'))

        cube = xr.open_zarr(self.CUBE_PATH)
        expected = np.array(['2019-01-01T14:00', '2019-01-02T12:00',
                             '2019-01-03T12:00', '2019-01-04T12:00',
                             '2019-01-05T12:00', '2019-01-06T12:00',
                             '2019-01-06T14:00', '2019-01-07T12:00',
                             '2019-01-08T12:00', '2019-01-09T12:00',
                             '2019-01-10T12:00', '2019-01-10T14:00',
                             '2019-01-11T12:00'], dtype=cube.time.dtype)
        actual = cube.time.values
        print(actual)
        self.assertEqual(len(expected), len(actual))
        self.assertEqual(None, cube.time.chunks)
        np.testing.assert_equal(actual, expected)

    def test_replace_time_slice(self):
        self.write_cube('2019-01-02', 10)

        replace_time_slice(self.CUBE_PATH, 5, self.make_slice('2019-01-06T02:00'))
        replace_time_slice(self.CUBE_PATH, 9, self.make_slice('2019-01-11T02:00'))
        replace_time_slice(self.CUBE_PATH, 0, self.make_slice('2019-01-01T02:00'))

        cube = xr.open_zarr(self.CUBE_PATH)
        expected = np.array(['2019-01-01T14:00', '2019-01-03T12:00',
                             '2019-01-04T12:00', '2019-01-05T12:00',
                             '2019-01-06T12:00', '2019-01-06T14:00',
                             '2019-01-08T12:00', '2019-01-09T12:00',
                             '2019-01-10T12:00', '2019-01-11T14:00'], dtype=cube.time.dtype)
        self.assertEqual(10, cube.time.size)
        self.assertEqual(None, cube.time.chunks)
        np.testing.assert_equal(cube.time.values, expected)

    def test_update_corrupt_cube(self):
        self.write_cube('2019-01-01', 3)

        cube = xr.open_zarr(self.CUBE_PATH)
        t, y, x = cube.precipitation.shape
        new_shape = y, t, x
        t, y, x = cube.precipitation.dims
        new_dims = y, t, x
        cube['precipitation'] = xr.DataArray(cube.precipitation.values.reshape(new_shape),
                                             dims=new_dims,
                                             coords=cube.precipitation.coords)
        cube.to_zarr(self.CUBE_PATH_2)

        with self.assertRaises(ValueError) as cm:
            insert_time_slice(self.CUBE_PATH_2, 2, self.make_slice('2019-01-02T06:30'))
        self.assertEqual("dimension 'time' of variable 'precipitation' must be first dimension",
                         f"{cm.exception}")


class ZarrStoreTest(unittest.TestCase):
    CUBE_PATH = 'store-test-cube.zarr'

    def setUp(self) -> None:
        rimraf(self.CUBE_PATH)

    def tearDown(self) -> None:
        rimraf(self.CUBE_PATH)

    def test_local(self):
        cube = new_cube(time_periods=10, time_start='2019-01-01',
                        variables=dict(precipitation=0.1,
                                       temperature=270.5,
                                       soil_moisture=0.2))
        cube = chunk_dataset(cube, dict(time=1, lat=90, lon=90), format_name='zarr')
        cube.to_zarr(self.CUBE_PATH)
        cube.close()

        diagnostic_store = DiagnosticStore(zarr.DirectoryStore(self.CUBE_PATH),
                                           logging_observer(log_path='local-cube.log'))
        xr.open_zarr(diagnostic_store)

    @unittest.skipUnless(False, 'is enabled')
    def test_remote(self):
        import s3fs
        endpoint_url = "https://s3.eu-central-1.amazonaws.com"
        s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(endpoint_url=endpoint_url))
        s3_store = s3fs.S3Map(root="xcube-examples/OLCI-SNS-RAW-CUBE-2.zarr", s3=s3, check=False)
        diagnostic_store = DiagnosticStore(s3_store, logging_observer(log_path='remote-cube.log'))
        xr.open_zarr(diagnostic_store)
