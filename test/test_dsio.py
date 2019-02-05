import unittest
from typing import Set

import numpy as np
import pandas as pd
import xarray as xr

from test.sampledata import new_test_dataset
from xcube.dsio import DatasetIO, MemDatasetIO, Netcdf4DatasetIO, ZarrDatasetIO, find_dataset_io, query_dataset_io, \
    CsvDatasetIO


# noinspection PyAbstractClass
class MyDatasetIO(DatasetIO):

    @property
    def name(self) -> str:
        return "test"

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
        ds_io.datasets['test.nc'] = ds1
        ds2 = ds_io.read('test.nc')
        self.assertIs(ds2, ds1)

    def test_write(self):
        ds_io = MemDatasetIO()
        ds1 = xr.Dataset()
        ds_io.write(ds1, 'test.nc')
        ds2 = ds_io.datasets['test.nc']
        self.assertIs(ds2, ds1)

    def test_append(self):
        ds_io = MemDatasetIO()
        ds1 = new_test_dataset('2017-02-01', 180, temperature=1.2, precipitation=2.1)
        ds2 = new_test_dataset('2017-02-02', 180, temperature=2.3, precipitation=3.2)
        ds3 = new_test_dataset('2017-02-03', 180, temperature=3.4, precipitation=4.3)
        ds_io.append(ds1, 'test.nc')
        ds_io.append(ds2, 'test.nc')
        ds_io.append(ds3, 'test.nc')
        ds4 = ds_io.datasets.get('test.nc')
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
