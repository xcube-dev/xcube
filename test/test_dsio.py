import unittest
from typing import Set

import numpy as np
import pandas as pd
import xarray as xr

from test.sampledata import new_test_dataset
from xcube.dsio import DatasetIO, MemDatasetIO


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
