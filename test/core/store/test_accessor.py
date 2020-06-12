import unittest

from xcube.core.store.accessor import find_data_opener_extensions
from xcube.core.store.accessor import find_data_writer_extensions


class ExtensionRegistryTest(unittest.TestCase):

    def test_find_data_opener_extensions(self):
        actual_ext = set(ext.name for ext in find_data_opener_extensions())
        self.assertIn('dataset:netcdf:posix', actual_ext)
        self.assertIn('dataset:zarr:posix', actual_ext)
        self.assertIn('dataset:zarr:s3', actual_ext)
        self.assertIn('geodataframe:shapefile:posix', actual_ext)
        self.assertIn('geodataframe:geojson:posix', actual_ext)

    def test_find_data_writer_extensions(self):
        actual_ext = set(ext.name for ext in find_data_writer_extensions())
        self.assertIn('dataset:netcdf:posix', actual_ext)
        self.assertIn('dataset:zarr:posix', actual_ext)
        self.assertIn('dataset:zarr:s3', actual_ext)
        self.assertIn('geodataframe:shapefile:posix', actual_ext)
        self.assertIn('geodataframe:geojson:posix', actual_ext)
