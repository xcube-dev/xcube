import unittest

from xcube.core.store import DataStoreError
from xcube.core.store import find_data_opener_extensions
from xcube.core.store import find_data_writer_extensions
from xcube.core.store import get_data_accessor_predicate
from xcube.util.extension import Extension


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

    def test_data_accessor_predicate(self):
        def ext(name: str) -> Extension:
            return Extension(point='test', name=name, component=object())

        p = get_data_accessor_predicate()
        self.assertEqual(True, p(ext('dataset:zarr:s3')))

        p = get_data_accessor_predicate(type_id='dataset')
        self.assertEqual(True, p(ext('dataset:zarr:s3')))
        self.assertEqual(False, p(ext('dataset[cube]:zarr:s3')))
        self.assertEqual(False, p(ext('mldataset:levels:s3')))
        self.assertEqual(False, p(ext('mldataset[cube]:levels:s3')))
        self.assertEqual(False, p(ext('geodataframe:geojson:posix')))

        p = get_data_accessor_predicate(type_id='mldataset[cube]')
        self.assertEqual(False, p(ext('dataset:zarr:s3')))
        self.assertEqual(False, p(ext('dataset[cube]:zarr:s3')))
        self.assertEqual(False, p(ext('mldataset:levels:s3')))
        self.assertEqual(True, p(ext('mldataset[cube]:levels:s3')))
        self.assertEqual(False, p(ext('geodataframe:geojson:posix')))

        p = get_data_accessor_predicate(format_id='levels')
        self.assertEqual(False, p(ext('dataset:zarr:s3')))
        self.assertEqual(False, p(ext('dataset[cube]:zarr:s3')))
        self.assertEqual(True, p(ext('mldataset:levels:s3')))
        self.assertEqual(True, p(ext('mldataset[cube]:levels:s3')))
        self.assertEqual(False, p(ext('geodataframe:geojson:posix')))

        p = get_data_accessor_predicate(storage_id='posix')
        self.assertEqual(False, p(ext('dataset:zarr:s3')))
        self.assertEqual(False, p(ext('mldataset:levels:s3')))
        self.assertEqual(True, p(ext('geodataframe:geojson:posix')))

        p = get_data_accessor_predicate(type_id='dataset')
        self.assertEqual(True, p(ext('*:*:memory')))

        p = get_data_accessor_predicate(format_id='levels')
        self.assertEqual(True, p(ext('*:*:memory')))

        p = get_data_accessor_predicate(storage_id='memory')
        self.assertEqual(True, p(ext('*:*:memory')))

        p = get_data_accessor_predicate(storage_id='posix')
        self.assertEqual(False, p(ext('*:*:memory')))

        with self.assertRaises(DataStoreError) as cm:
            p(ext('geodataframe,geojson:posix'))
        self.assertEqual('Illegal data opener/writer extension name "geodataframe,geojson:posix"', f'{cm.exception}')
