import unittest

import numpy as np
import xarray as xr
import zarr
from xcube.api import new_cube, chunk_dataset
from xcube.util.dsgrow import add_time_slice
from xcube.util.dsio import rimraf

from .diagnosticstore import DiagnosticStore, logging_observer

import time

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
                                             temperature=275.8,
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
                                             temperature=275.8,
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
