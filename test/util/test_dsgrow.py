import time
import unittest

import numpy as np
import xarray as xr
import zarr

from xcube.api import chunk_dataset, new_cube
from xcube.util.dsgrow import add_time_slice, get_time_insert_index
from xcube.util.dsio import rimraf
from .diagnosticstore import DiagnosticStore, logging_observer


class AddTimeSliceTest(unittest.TestCase):
    CUBE_PATH = 'test-cube.zarr'

    def setUp(self) -> None:
        rimraf(self.CUBE_PATH)

    def tearDown(self) -> None:
        rimraf(self.CUBE_PATH)

    def test_append_time_slice(self):

        times = []
        for m in range(1, 4):
            for d in range(1, 10):
                times.append(f'2019-0{m}-0{d}T00:00:00')

        time_avg = 0
        for i_time in range(len(times)):
            time_slice = new_cube(time_periods=1, time_start=times[i_time],
                                  variables=dict(precipitation=0.1 * i_time,
                                                 temperature=270 + 0.5 * i_time,
                                                 soil_moisture=0.2 * i_time))
            t0 = time.perf_counter()
            add_time_slice(self.CUBE_PATH, time_slice, dict(time=1, lat=90, lon=90))
            time_avg += time.perf_counter() - t0
        time_avg /= len(times)
        print(f'add_time_slice() average append time: {time_avg} seconds')

        with xr.open_zarr(self.CUBE_PATH) as cube:
            self.assertEqual(len(times), cube.time.size)
            self.assertEqual(None, cube.time.chunks)
            actual = cube.time.values
            expected = np.array(['2019-01-01T12:00', '2019-01-02T12:00',
                                 '2019-01-03T12:00', '2019-01-04T12:00',
                                 '2019-01-05T12:00', '2019-01-06T12:00',
                                 '2019-01-07T12:00', '2019-01-08T12:00',
                                 '2019-01-09T12:00', '2019-02-01T12:00',
                                 '2019-02-02T12:00', '2019-02-03T12:00',
                                 '2019-02-04T12:00', '2019-02-05T12:00',
                                 '2019-02-06T12:00', '2019-02-07T12:00',
                                 '2019-02-08T12:00', '2019-02-09T12:00',
                                 '2019-03-01T12:00', '2019-03-02T12:00',
                                 '2019-03-03T12:00', '2019-03-04T12:00',
                                 '2019-03-05T12:00', '2019-03-06T12:00',
                                 '2019-03-07T12:00', '2019-03-08T12:00',
                                 '2019-03-09T12:00'], dtype=actual.dtype)
            np.testing.assert_equal(actual, expected)

            actual = cube.temperature.isel(lat=0, lon=0).values
            expected = np.array([270., 270.5, 271., 271.5, 272., 272.5, 273., 273.5, 274.,
                                 274.5, 275., 275.5, 276., 276.5, 277., 277.5, 278., 278.5,
                                 279., 279.5, 280., 280.5, 281., 281.5, 282., 282.5, 283.],
                                dtype=actual.dtype)
            np.testing.assert_almost_equal(actual, expected)

    def test_get_time_insert_index(self):
        cube = new_cube(time_periods=10, time_start='2019-01-01',
                        variables=dict(precipitation=0.1,
                                       temperature=270.5,
                                       soil_moisture=0.2))
        chunk_sizes = dict(time=1, lat=90, lon=90)
        cube = chunk_dataset(cube, chunk_sizes, format_name='zarr')
        cube.to_zarr(self.CUBE_PATH)
        cube.close()
        insert_index = get_time_insert_index(self.CUBE_PATH, np.datetime64('2019-01-02T00:00:00'))
        self.assertEqual(1, insert_index)
        with self.assertRaises(NotImplementedError) as cm:
            get_time_insert_index(self.CUBE_PATH, np.datetime64('2019-01-01T12:00:00'))
        self.assertEqual('time already found in test-cube.zarr, this is not yet supported', f"{cm.exception}")
        insert_index = get_time_insert_index(self.CUBE_PATH, np.datetime64('2019-01-12T00:00:00'))
        self.assertEqual(-1, insert_index)

    def test_insert_time_slice(self):
        cube = new_cube(time_periods=10, time_start='2019-01-01',
                        variables=dict(precipitation=0.1,
                                       temperature=270.5,
                                       soil_moisture=0.2))
        chunk_sizes = dict(time=1, lat=90, lon=90)
        cube = chunk_dataset(cube, chunk_sizes, format_name='zarr')
        cube.to_zarr(self.CUBE_PATH)
        cube.close()

        time_slice = new_cube(time_periods=1, time_start='2019-01-08T16:30:00',
                              variables=dict(precipitation=0.2,
                                             temperature=274.8,
                                             soil_moisture=0.4))

        t0 = time.perf_counter()
        add_time_slice(self.CUBE_PATH, time_slice, chunk_sizes)
        print(f'add_time_slice() insert time: {time.perf_counter() - t0} seconds')

        with xr.open_zarr(self.CUBE_PATH) as cube:
            self.assertEqual(11, cube.time.size)
            self.assertEqual(None, cube.time.chunks)
            actual = cube.time.values
            expected = np.array(['2019-01-01T12:00', '2019-01-02T12:00',
                                 '2019-01-03T12:00', '2019-01-04T12:00',
                                 '2019-01-05T12:00', '2019-01-06T12:00',
                                 '2019-01-07T12:00', '2019-01-08T12:00',
                                 '2019-01-09T04:30', '2019-01-09T12:00',
                                 '2019-01-10T12:00'], dtype=actual.dtype)
            np.testing.assert_equal(actual, expected)

        time_slice = new_cube(time_periods=1, time_start='2018-12-31T10:15:00',
                              variables=dict(precipitation=0.2,
                                             temperature=273.8,
                                             soil_moisture=0.4))

        t0 = time.perf_counter()
        add_time_slice(self.CUBE_PATH, time_slice, chunk_sizes)
        print(f'add_time_slice() prepend time: {time.perf_counter() - t0} seconds')

        with xr.open_zarr(self.CUBE_PATH) as cube:
            self.assertEqual(12, cube.time.size)
            self.assertEqual(None, cube.time.chunks)
            actual = cube.time.values
            expected = np.array(['2018-12-31T22:15',
                                 '2019-01-01T12:00', '2019-01-02T12:00',
                                 '2019-01-03T12:00', '2019-01-04T12:00',
                                 '2019-01-05T12:00', '2019-01-06T12:00',
                                 '2019-01-07T12:00', '2019-01-08T12:00',
                                 '2019-01-09T04:30', '2019-01-09T12:00',
                                 '2019-01-10T12:00'], dtype=actual.dtype)
            np.testing.assert_equal(actual, expected)

            actual = cube.temperature.isel(lat=0, lon=0).values
            expected = np.array([273.8, 270.5, 270.5, 270.5, 270.5, 270.5, 270.5, 270.5, 270.5, 274.8, 270.5, 270.5],
                                dtype=actual.dtype)
            np.testing.assert_almost_equal(actual, expected)

    def test_insert_time_slice_into_corrupt_cube(self):
        cube = new_cube(time_periods=10, time_start='2019-01-01',
                        variables=dict(precipitation=0.1,
                                       temperature=270.5,
                                       soil_moisture=0.2))
        chunk_sizes = dict(time=1, lat=90, lon=90)
        cube = chunk_dataset(cube, chunk_sizes, format_name='zarr')
        t, y, x = cube.precipitation.shape
        new_shape = y, t, x
        t, y, x = cube.precipitation.dims
        new_dims = y, t, x
        false_cube = xr.DataArray(cube.precipitation.values.reshape(new_shape), dims=new_dims,
                                  coords=cube.precipitation.coords)
        false_cube = false_cube.to_dataset(dim='precipitation')
        false_cube.to_zarr(self.CUBE_PATH)
        false_cube.close()
        cube.close()
        time_slice = new_cube(time_periods=1, time_start='2019-01-08T16:30:00',
                              variables=dict(precipitation=0.2,
                                             temperature=274.8,
                                             soil_moisture=0.4))
        with self.assertRaises(ValueError) as cm:
            add_time_slice(self.CUBE_PATH, time_slice)
        self.assertEqual("Variable: precipitation Dimension 'time' must be first dimension", f"{cm.exception}")


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
        endpoint_url = "http://obs.eu-de.otc.t-systems.com"
        s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(endpoint_url=endpoint_url))
        s3_store = s3fs.S3Map(root="cyanoalert/cyanoalert-olci-lswe-l2c-v1.zarr", s3=s3, check=False)
        diagnostic_store = DiagnosticStore(s3_store, logging_observer(log_path='remote-cube.log'))
        xr.open_zarr(diagnostic_store)
