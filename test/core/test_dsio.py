import os
import unittest
from typing import Set

import boto3
import fsspec
import numpy as np
import pandas as pd
import xarray as xr
import moto

from test.sampledata import new_test_dataset
from xcube.core.dsio import CsvDatasetIO, DatasetIO, MemDatasetIO, Netcdf4DatasetIO, ZarrDatasetIO, find_dataset_io, \
    query_dataset_io, _get_path_or_store, write_cube
from xcube.core.dsio import open_dataset, write_dataset
from xcube.core.new import new_cube

TEST_NC_FILE = "test.nc"
TEST_NC_FILE_2 = "test-2.nc"


class OpenWriteDatasetTest(unittest.TestCase):
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

    def test_write_dataset(self):

        dataset = new_cube()
        try:
            write_dataset(dataset, TEST_NC_FILE_2)
            self.assertTrue(os.path.isfile(TEST_NC_FILE_2))
        finally:
            if os.path.isfile(TEST_NC_FILE_2):
                os.remove(TEST_NC_FILE_2)


# noinspection PyAbstractClass
class MyDatasetIO(DatasetIO):

    def __init__(self):
        super().__init__("test")

    @property
    def description(self) -> str:
        return "test"

    @property
    def ext(self) -> str:
        return "test"

    @property
    def modes(self) -> Set[str]:
        return set()

    def fitness(self, input_path: str, path_type: str = None) -> float:
        return 0.0


class DatasetIOTest(unittest.TestCase):

    def test_read_raises(self):
        ds_io = MyDatasetIO()
        with self.assertRaises(NotImplementedError):
            ds_io.read('test.nc')

    def test_write_raises(self):
        ds_io = MyDatasetIO()
        with self.assertRaises(NotImplementedError):
            # noinspection PyTypeChecker
            ds_io.write(None, 'test.nc')

    def test_append_raises(self):
        ds_io = MyDatasetIO()
        with self.assertRaises(NotImplementedError):
            # noinspection PyTypeChecker
            ds_io.append(None, 'test.nc')

    def test_insert_raises(self):
        ds_io = MyDatasetIO()
        with self.assertRaises(NotImplementedError):
            # noinspection PyTypeChecker
            ds_io.insert(None, 0, 'test.nc')

    def test_replace_raises(self):
        ds_io = MyDatasetIO()
        with self.assertRaises(NotImplementedError):
            # noinspection PyTypeChecker
            ds_io.replace(None, 0, 'test.nc')

    def test_update_raises(self):
        ds_io = MyDatasetIO()
        with self.assertRaises(NotImplementedError):
            # noinspection PyTypeChecker
            ds_io.update(None, 'test.nc')


class MemDatasetIOTest(unittest.TestCase):

    def test_props(self):
        ds_io = MemDatasetIO()
        self.assertEqual('mem', ds_io.name)
        self.assertEqual('mem', ds_io.ext)
        self.assertEqual('In-memory dataset I/O', ds_io.description)
        self.assertEqual({'r', 'w', 'a'}, ds_io.modes)

    def test_fitness(self):
        ds_io = MemDatasetIO()

        self.assertEqual(0.75, ds_io.fitness("test.mem", path_type=None))
        self.assertEqual(0.75, ds_io.fitness("test.mem", path_type="file"))
        self.assertEqual(0.75, ds_io.fitness("test.mem", path_type="dir"))
        self.assertEqual(0.75, ds_io.fitness("http://dsio/test.mem", path_type="url"))

        self.assertEqual(0.0, ds_io.fitness("test.png", path_type=None))
        self.assertEqual(0.0, ds_io.fitness("test.png", path_type="file"))
        self.assertEqual(0.0, ds_io.fitness("test.png", path_type="dir"))
        self.assertEqual(0.0, ds_io.fitness("http://dsio/test.png", path_type="url"))

        ds_io.write(xr.Dataset(), "bibo.odod")
        self.assertEqual(1.0, ds_io.fitness("bibo.odod", path_type=None))
        self.assertEqual(1.0, ds_io.fitness("bibo.odod", path_type="file"))
        self.assertEqual(1.0, ds_io.fitness("bibo.odod", path_type="dir"))
        self.assertEqual(1.0, ds_io.fitness("bibo.odod", path_type="url"))

    def test_read(self):
        ds_io = MemDatasetIO()
        with self.assertRaises(FileNotFoundError):
            ds_io.read('test.nc')
        ds1 = xr.Dataset()
        ds_io._datasets['test.nc'] = ds1
        ds2 = ds_io.read('test.nc')
        self.assertIs(ds2, ds1)

    def test_write(self):
        ds_io = MemDatasetIO()
        ds1 = xr.Dataset()
        ds_io.write(ds1, 'test.nc')
        ds2 = ds_io._datasets['test.nc']
        self.assertIs(ds2, ds1)

    def test_append(self):
        ds_io = MemDatasetIO()
        ds1 = new_test_dataset('2017-02-01', 180, temperature=1.2, precipitation=2.1)
        ds2 = new_test_dataset('2017-02-02', 180, temperature=2.3, precipitation=3.2)
        ds3 = new_test_dataset('2017-02-03', 180, temperature=3.4, precipitation=4.3)
        ds_io.append(ds1, 'test.nc')
        ds_io.append(ds2, 'test.nc')
        ds_io.append(ds3, 'test.nc')
        ds4 = ds_io._datasets.get('test.nc')
        self.assertIsNotNone(ds4)
        self.assertIn('time', ds4)
        self.assertIn('temperature', ds4)
        self.assertEqual(('time', 'lat', 'lon'), ds4.temperature.dims)
        self.assertEqual((3, 180, 360), ds4.temperature.shape)
        expected_time = xr.DataArray(pd.to_datetime(['2017-02-01', '2017-02-02', '2017-02-03']))
        np.testing.assert_equal(expected_time.values, ds4.time.values)


class Netcdf4DatasetIOTest(unittest.TestCase):

    def test_props(self):
        ds_io = Netcdf4DatasetIO()
        self.assertEqual('netcdf4', ds_io.name)
        self.assertEqual('nc', ds_io.ext)
        self.assertEqual('NetCDF-4 file format', ds_io.description)
        self.assertEqual({'a', 'r', 'w'}, ds_io.modes)

    def test_fitness(self):
        ds_io = Netcdf4DatasetIO()

        self.assertEqual(0.875, ds_io.fitness("test.nc", path_type=None))
        self.assertEqual(0.875, ds_io.fitness("test.hdf", path_type=None))
        self.assertEqual(0.875, ds_io.fitness("test.h5", path_type=None))

        self.assertEqual(1.0, ds_io.fitness("test.nc", path_type="file"))
        self.assertEqual(1.0, ds_io.fitness("test.hdf", path_type="file"))
        self.assertEqual(1.0, ds_io.fitness("test.h5", path_type="file"))

        self.assertEqual(0.0, ds_io.fitness("test.nc", path_type="dir"))
        self.assertEqual(0.0, ds_io.fitness("test.hdf", path_type="dir"))
        self.assertEqual(0.0, ds_io.fitness("test.h5", path_type="dir"))

        self.assertEqual(0.0, ds_io.fitness("https://dsio/test.nc", path_type="url"))
        self.assertEqual(0.0, ds_io.fitness("https://dsio/test.hdf", path_type="url"))
        self.assertEqual(0.0, ds_io.fitness("https://dsio/test.h5", path_type="url"))

        self.assertEqual(0.125, ds_io.fitness("test.tif", path_type=None))
        self.assertEqual(0.25, ds_io.fitness("test.tif", path_type="file"))
        self.assertEqual(0.0, ds_io.fitness("test.tif", path_type="dir"))
        self.assertEqual(0.0, ds_io.fitness("https://dsio/test.tif", path_type="url"))

    def test_read(self):
        ds_io = Netcdf4DatasetIO()
        with self.assertRaises(FileNotFoundError):
            ds_io.read('test.nc')


class ZarrDatasetIOTest(unittest.TestCase):

    def test_props(self):
        ds_io = ZarrDatasetIO()
        self.assertEqual('zarr', ds_io.name)
        self.assertEqual('zarr', ds_io.ext)
        self.assertEqual('Zarr file format (http://zarr.readthedocs.io)', ds_io.description)
        self.assertEqual({'a', 'r', 'w'}, ds_io.modes)

    def test_fitness(self):
        ds_io = ZarrDatasetIO()

        self.assertEqual(0.875, ds_io.fitness("test.zarr", path_type=None))
        self.assertEqual(0.0, ds_io.fitness("test.zarr", path_type="file"))
        self.assertEqual(1.0, ds_io.fitness("test.zarr", path_type="dir"))
        self.assertEqual(0.875, ds_io.fitness("http://dsio/test.zarr", path_type="url"))

        self.assertEqual(0.875, ds_io.fitness("test.zarr.zip", path_type=None))
        self.assertEqual(1.0, ds_io.fitness("test.zarr.zip", path_type="file"))
        self.assertEqual(0.0, ds_io.fitness("test.zarr.zip", path_type="dir"))
        self.assertEqual(0.0, ds_io.fitness("http://dsio/test.zarr.zip", path_type="url"))

        self.assertEqual(0.0, ds_io.fitness("test.png", path_type=None))
        self.assertEqual(0.0, ds_io.fitness("test.png", path_type="file"))
        self.assertEqual(0.25, ds_io.fitness("test.png", path_type="dir"))
        self.assertEqual(0.125, ds_io.fitness("http://dsio/test.png", path_type="url"))

    def test_read(self):
        ds_io = ZarrDatasetIO()
        with self.assertRaises(ValueError):
            ds_io.read('test.zarr')


class CsvDatasetIOTest(unittest.TestCase):

    def test_props(self):
        ds_io = CsvDatasetIO()
        self.assertEqual('csv', ds_io.name)
        self.assertEqual('csv', ds_io.ext)
        self.assertEqual('CSV file format', ds_io.description)
        self.assertEqual({'r', 'w'}, ds_io.modes)

    def test_fitness(self):
        ds_io = CsvDatasetIO()
        self.assertEqual(0.875, ds_io.fitness("test.csv", path_type=None))
        self.assertEqual(1.0, ds_io.fitness("test.csv", path_type="file"))
        self.assertEqual(0.0, ds_io.fitness("test.csv", path_type="dir"))
        self.assertEqual(0.875, ds_io.fitness("http://dsio/test.csv", path_type="url"))

    def test_read(self):
        ds_io = CsvDatasetIO()
        with self.assertRaises(FileNotFoundError):
            ds_io.read('test.csv')


class FindDatasetIOTest(unittest.TestCase):

    def test_find_by_name(self):
        ds_io = find_dataset_io('netcdf4')
        self.assertIsInstance(ds_io, Netcdf4DatasetIO)

        ds_io = find_dataset_io('zarr', modes=['a'])
        self.assertIsInstance(ds_io, ZarrDatasetIO)
        ds_io = find_dataset_io('zarr', modes=['w'])
        self.assertIsInstance(ds_io, ZarrDatasetIO)
        ds_io = find_dataset_io('zarr', modes=['r'])
        self.assertIsInstance(ds_io, ZarrDatasetIO)

        ds_io = find_dataset_io('mem')
        self.assertIsInstance(ds_io, MemDatasetIO)

        ds_io = find_dataset_io('bibo', default=MemDatasetIO())
        self.assertIsInstance(ds_io, MemDatasetIO)

    def test_find_by_ext(self):
        ds_io = find_dataset_io('nc')
        self.assertIsInstance(ds_io, Netcdf4DatasetIO)


class QueryDatasetIOsTest(unittest.TestCase):
    def test_query_dataset_io(self):
        ds_ios = query_dataset_io()
        self.assertEqual(4, len(ds_ios))

    def test_query_dataset_io_with_fn(self):
        ds_ios = query_dataset_io(lambda ds_io: ds_io.name == 'mem')
        self.assertEqual(1, len(ds_ios))
        self.assertIsInstance(ds_ios[0], MemDatasetIO)


class ContextManagerTest(unittest.TestCase):
    def test_it(self):
        class A:
            def __init__(self):
                self.closed = False

            def __enter__(self):
                return self

            def __exit__(self, exception_type, exception_value, traceback):
                self.close()

            def close(self):
                self.closed = True

        def open_a():
            return A()

        with open_a() as a:
            pass

        self.assertEqual(True, a.closed)

        a = open_a()
        self.assertIsInstance(a, A)
        self.assertEqual(False, a.closed)


class GetPathOrStoreTest(unittest.TestCase):
    def test_path_or_store_read_from_bucket(self):
        path = _get_path_or_store(root=None,
                                  path="http://obs.eu-de.otc.t-systems.com/dcs4cop-obs-02/OLCI-SNS-RAW-CUBE-2.zarr",
                                  mode="read", client_kwargs=None)[0]
        self.assertIsInstance(path, fsspec.mapping.FSMap)

    def test_path_or_store_write_to_bucket(self):
        path = _get_path_or_store(root=None,
                                  path="http://obs.eu-de.otc.t-systems.com/fake_bucket/fake_cube.zarr",
                                  mode="write",
                                  client_kwargs={"aws_access_key_id": "some_fake_id",
                                                 "aws_secret_access_key": "some_fake_key"})[0]
        self.assertIsInstance(path, fsspec.mapping.FSMap)

    def test_path_or_store_read_from_local(self):
        path = _get_path_or_store(root=None,
                                  path="../examples/serve/demo/cube-1-250-250.zarr",
                                  mode="read", client_kwargs=None)[0]
        self.assertIsInstance(path, str)

    def test_path_or_store_write_to_bucket_env_credentials(self):
        os.environ['aws_access_key_id'] = 'some_fake_id'
        os.environ['aws_secret_access_key'] = 'some_fake_key'
        path = _get_path_or_store(root=None,
                                  path="http://obs.eu-de.otc.t-systems.com/fake_bucket/fake_cube.zarr",
                                  mode="write", client_kwargs=None)[0]
        self.assertIsInstance(path, fsspec.mapping.FSMap)
        del os.environ['aws_access_key_id']
        del os.environ['aws_secret_access_key']


class TestUploadToS3Bucket(unittest.TestCase):

    def test_upload_to_s3(self):
        with moto.mock_s3():
            s3_conn = boto3.client('s3')
            s3_conn.create_bucket(Bucket='upload_bucket', ACL="public-read")
            client_kwargs = {"aws_access_key_id": "test_fake_id", "aws_secret_access_key": "test_fake_secret"}
            ds1 = xr.open_zarr("examples/serve/demo/cube-1-250-250.zarr")
            write_cube(ds1, "https://s3.amazonaws.com/upload_bucket/cube-1-250-250.zarr", "zarr",
                       client_kwargs=client_kwargs)
            self.assertIn('cube-1-250-250.zarr/.zattrs',
                          s3_conn.list_objects(Bucket='upload_bucket')["Contents"][0]["Key"])
