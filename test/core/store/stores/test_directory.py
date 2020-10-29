import os.path
import unittest

from xcube.core.store import new_data_store
from xcube.core.store import TYPE_ID_CUBE
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
        self.assertEqual({'dataset', 'dataset[multilevel]', 'geodataframe'},
                         set(self.store.get_type_ids()))

    def test_get_data_opener_ids(self):
        self.assertEqual({'dataset:netcdf:posix',
                          'dataset:zarr:posix',
                          'geodataframe:geojson:posix',
                          'geodataframe:shapefile:posix'},
                         set(self.store.get_data_opener_ids()))
        self.assertEqual({'dataset:netcdf:posix',
                          'dataset:zarr:posix',
                          'geodataframe:geojson:posix',
                          'geodataframe:shapefile:posix'},
                         set(self.store.get_data_opener_ids(type_id='*')))
        self.assertEqual({'dataset:netcdf:posix',
                          'dataset:zarr:posix'},
                         set(self.store.get_data_opener_ids(type_id='dataset')))
        with self.assertRaises(ValueError) as cm:
            set(self.store.get_data_opener_ids(type_id='dataset[cube]'))
        self.assertEqual("type_id must be one of ('dataset', 'dataset[multilevel]', 'geodataframe')", f'{cm.exception}')
        self.assertEqual(set(),
                         set(self.store.get_data_opener_ids(type_id='dataset[multilevel]')))
        self.assertEqual({'geodataframe:geojson:posix',
                          'geodataframe:shapefile:posix'},
                         set(self.store.get_data_opener_ids(type_id='geodataframe')))

    def test_get_data_writer_ids(self):
        self.assertEqual({'dataset:netcdf:posix',
                          'dataset:zarr:posix',
                          'geodataframe:geojson:posix',
                          'geodataframe:shapefile:posix'},
                         set(self.store.get_data_writer_ids()))
        self.assertEqual({'dataset:netcdf:posix',
                          'dataset:zarr:posix',
                          'geodataframe:geojson:posix',
                          'geodataframe:shapefile:posix'},
                         set(self.store.get_data_writer_ids(type_id='*')))
        self.assertEqual({'dataset:netcdf:posix',
                          'dataset:zarr:posix'},
                         set(self.store.get_data_writer_ids(type_id='dataset')))
        with self.assertRaises(ValueError) as cm:
            set(self.store.get_data_writer_ids(type_id='dataset[cube]'))
        self.assertEqual("type_id must be one of ('dataset', 'dataset[multilevel]', 'geodataframe')", f'{cm.exception}')
        self.assertEqual(set(),
                         set(self.store.get_data_writer_ids(type_id='dataset[multilevel]')))
        self.assertEqual({'geodataframe:geojson:posix',
                          'geodataframe:shapefile:posix'},
                         set(self.store.get_data_writer_ids(type_id='geodataframe')))

    def test_get_data_ids(self):
        self.assertEqual(
            {
                ('cube-1-250-250.zarr', None),
                ('cube-5-100-200.zarr', None),
                ('cube.nc', None),
            },
            set(self.store.get_data_ids()))
        self.assertEqual(
            {
                ('cube-1-250-250.zarr', None),
                ('cube-5-100-200.zarr', None),
                ('cube.nc', None),
            },
            set(self.store.get_data_ids('*')))
        self.assertEqual(
            {
                ('cube-1-250-250.zarr', None),
                ('cube-5-100-200.zarr', None),
                ('cube.nc', None),
            },
            set(self.store.get_data_ids('dataset')))
        self.assertEqual(
            set(),
            set(self.store.get_data_ids('dataset[multilevel]')))
        with self.assertRaises(ValueError) as cm:
            set(self.store.get_data_ids(type_id='dataset[cube]'))
        self.assertEqual("type_id must be one of ('dataset', 'dataset[multilevel]', 'geodataframe')", f'{cm.exception}')

    def test_search_data(self):
        result = list(self.store.search_data())
        self.assertEqual(3, len(result))
        self.assertEqual(result[0].data_id, 'cube-1-250-250.zarr')
        self.assertEqual(result[0].type_id, TYPE_ID_CUBE)
        self.assertEqual(result[1].data_id, 'cube-5-100-200.zarr')
        self.assertEqual(result[1].type_id, TYPE_ID_CUBE)
        self.assertEqual(result[2].data_id, 'cube.nc')
        self.assertEqual(result[2].type_id, TYPE_ID_CUBE)

        result = list(self.store.search_data('*'))
        self.assertEqual(3, len(result))
        self.assertEqual(result[0].data_id, 'cube-1-250-250.zarr')
        self.assertEqual(result[0].type_id, TYPE_ID_CUBE)
        self.assertEqual(result[1].data_id, 'cube-5-100-200.zarr')
        self.assertEqual(result[1].type_id, TYPE_ID_CUBE)
        self.assertEqual(result[2].data_id, 'cube.nc')
        self.assertEqual(result[2].type_id, TYPE_ID_CUBE)

        result = list(self.store.search_data(type_id='dataset'))
        self.assertEqual(3, len(result))
        self.assertEqual(result[0].data_id, 'cube-1-250-250.zarr')
        self.assertEqual(result[0].type_id, TYPE_ID_CUBE)
        self.assertEqual(result[1].data_id, 'cube-5-100-200.zarr')
        self.assertEqual(result[1].type_id, TYPE_ID_CUBE)
        self.assertEqual(result[2].data_id, 'cube.nc')
        self.assertEqual(result[2].type_id, TYPE_ID_CUBE)

        with self.assertRaises(ValueError) as cm:
            list(self.store.search_data(type_id='dataset[cube]'))
        self.assertEqual("type_id must be one of ('dataset', 'dataset[multilevel]', 'geodataframe')", f'{cm.exception}')

        result = list(self.store.search_data(type_id='geodataframe'))
        self.assertEqual(0, len(result))
