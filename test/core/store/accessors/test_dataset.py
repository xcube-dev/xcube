import unittest

from xcube.core.store.accessor import new_data_opener
from xcube.core.store.accessor import new_data_writer
from xcube.core.store.accessors.dataset import DatasetNetcdfPosixDataAccessor
from xcube.core.store.accessors.dataset import DatasetZarrPosixAccessor
from xcube.core.store.accessors.dataset import DatasetZarrS3Accessor


class DatasetZarrPosixAccessorTest(unittest.TestCase):

    def test_new_data_opener(self):
        opener = new_data_opener('dataset:zarr:posix')
        self.assertIsInstance(opener, DatasetZarrPosixAccessor)

    def test_new_data_writer(self):
        writer = new_data_writer('dataset:zarr:posix')
        self.assertIsInstance(writer, DatasetZarrPosixAccessor)


class DatasetZarrS3AccessorAccessorTest(unittest.TestCase):

    def test_new_data_opener(self):
        opener = new_data_opener('dataset:zarr:s3')
        self.assertIsInstance(opener, DatasetZarrS3Accessor)

    def test_new_data_writer(self):
        writer = new_data_writer('dataset:zarr:s3')
        self.assertIsInstance(writer, DatasetZarrS3Accessor)


class DatasetNetcdfPosixDataAccessorAccessorTest(unittest.TestCase):

    def test_new_data_opener(self):
        opener = new_data_opener('dataset:netcdf:posix')
        self.assertIsInstance(opener, DatasetNetcdfPosixDataAccessor)

    def test_new_data_writer(self):
        writer = new_data_writer('dataset:netcdf:posix')
        self.assertIsInstance(writer, DatasetNetcdfPosixDataAccessor)
