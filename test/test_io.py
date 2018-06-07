import unittest
from typing import Set

from xcube.io import DatasetIO


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
