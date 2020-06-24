import os.path
import unittest

from xcube.core.store import new_data_store
from xcube.core.store.stores.directory import DirectoryDataStore


class DirectoryDataStoreTest(unittest.TestCase):

    def setUp(self) -> None:
        self._store = new_data_store('directory',
                                     base_dir=os.path.join(os.path.dirname(__file__),
                                                           '..', '..', '..', '..', 'examples', 'serve', 'demo'))
        self.assertIsInstance(self.store, DirectoryDataStore)

    @property
    def store(self) -> DirectoryDataStore:
        # noinspection PyTypeChecker
        return self._store

    def test_get_data_store_params_schema(self):
        schema = self.store.get_data_store_params_schema()
        self.assertEqual(
            {'base_dir',
             'read_only'},
            set(schema.properties.keys())
        )
        self.assertEqual({'base_dir'}, schema.required)

    def test_get_open_data_params_schema(self):
        schema = self.store.get_open_data_params_schema()
        self.assertEqual(
            {'chunks',
             'consolidated',
             'decode_cf',
             'decode_coords',
             'decode_times',
             'drop_variables',
             'group',
             'mask_and_scale'},
            set(schema.properties.keys())
        )
        self.assertEqual(set(), schema.required)

    def test_get_write_data_params_schema(self):
        schema = self.store.get_write_data_params_schema()
        self.assertEqual(
            {'append_dim',
             'group',
             'consolidated',
             'encoding'},
            set(schema.properties.keys())
        )
        self.assertEqual(set(), schema.required)

    def test_get_type_ids(self):
        self.assertEqual({'geodataframe', 'mldataset', 'dataset'},
                         set(self.store.get_type_ids()))

    def test_get_data_opener_ids(self):
        self.assertEqual({'dataset:netcdf:posix',
                          'dataset:zarr:posix',
                          'geodataframe:geojson:posix',
                          'geodataframe:shapefile:posix'},
                         set(self.store.get_data_opener_ids()))

    def test_get_data_writer_ids(self):
        self.assertEqual({'dataset:netcdf:posix',
                          'dataset:zarr:posix',
                          'geodataframe:geojson:posix',
                          'geodataframe:shapefile:posix'},
                         set(self.store.get_data_writer_ids()))

    def test_get_data_ids(self):
        self.assertEqual(
            {
                'cube-1-250-250.zarr',
                'cube-5-100-200.zarr',
                'cube.nc'
            },
            set(self.store.get_data_ids()))
