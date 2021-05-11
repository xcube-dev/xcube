import json
import os.path
import shutil
import unittest

from xcube.core.new import new_cube
from xcube.core.store import DataStoreError, DatasetDescriptor
from xcube.core.store import TYPE_SPECIFIER_CUBE
from xcube.core.store import TYPE_SPECIFIER_DATASET
from xcube.core.store import new_data_store
from xcube.core.store.stores.directory import DirectoryDataStore
from xcube.util.jsonschema import JsonObjectSchema


class DirectoryDataStoreTest(unittest.TestCase):

    def setUp(self) -> None:
        base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'examples', 'serve', 'demo')
        self._store = new_data_store('directory', base_dir=base_dir)
        self.assertIsInstance(self.store, DirectoryDataStore)
        # noinspection PyUnresolvedReferences
        self.assertEqual(base_dir, self._store.base_dir)
        # noinspection PyUnresolvedReferences
        self.assertEqual(False, self._store.read_only)

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

    def test_get_search_params_schema(self):
        schema = self.store.get_search_params_schema()
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertEqual({}, schema.properties)
        self.assertEqual(False, schema.additional_properties)

        schema = self.store.get_search_params_schema(type_specifier='geodataframe')
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertEqual({}, schema.properties)
        self.assertEqual(False, schema.additional_properties)

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

    def test_get_type_specifiers(self):
        self.assertEqual({'dataset', 'dataset[multilevel]', 'geodataframe'},
                         set(self.store.get_type_specifiers()))

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
                         set(self.store.get_data_opener_ids(type_specifier='*')))
        self.assertEqual({'dataset:netcdf:posix',
                          'dataset:zarr:posix'},
                         set(self.store.get_data_opener_ids(type_specifier='dataset')))
        with self.assertRaises(ValueError) as cm:
            set(self.store.get_data_opener_ids(type_specifier='dataset[cube]'))
        self.assertEqual("type_specifier must be one of ('dataset', 'dataset[multilevel]', "
                         "'geodataframe')",
                         f'{cm.exception}')
        self.assertEqual(set(),
                         set(self.store.get_data_opener_ids(type_specifier='dataset[multilevel]')))
        self.assertEqual({'geodataframe:geojson:posix',
                          'geodataframe:shapefile:posix'},
                         set(self.store.get_data_opener_ids(type_specifier='geodataframe')))
        self.assertEqual({'dataset:netcdf:posix'},
                         set(self.store.get_data_opener_ids(data_id='dgdf.nc')))
        self.assertEqual({'dataset:zarr:posix'},
                         set(self.store.get_data_opener_ids(data_id='dgdf.zarr')))

    def test_get_type_specifiers_for_data(self):
        self.assertEqual(('dataset',),
                         self.store.get_type_specifiers_for_data('cube-1-250-250.zarr'))
        self.assertEqual(('dataset',), self.store.get_type_specifiers_for_data('cube.nc'))
        with self.assertRaises(DataStoreError) as cm:
            set(self.store.get_type_specifiers_for_data('xyz.levels'))
        self.assertEqual('Data resource "xyz.levels" does not exist in store',
                         f'{cm.exception}')

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
                         set(self.store.get_data_writer_ids(type_specifier='*')))
        self.assertEqual({'dataset:netcdf:posix',
                          'dataset:zarr:posix'},
                         set(self.store.get_data_writer_ids(type_specifier='dataset')))
        with self.assertRaises(ValueError) as cm:
            set(self.store.get_data_writer_ids(type_specifier='dataset[cube]'))
        self.assertEqual("type_specifier must be one of ('dataset', 'dataset[multilevel]', "
                         "'geodataframe')",
                         f'{cm.exception}')
        self.assertEqual(set(),
                         set(self.store.get_data_writer_ids(type_specifier='dataset[multilevel]')))
        self.assertEqual({'geodataframe:geojson:posix',
                          'geodataframe:shapefile:posix'},
                         set(self.store.get_data_writer_ids(type_specifier='geodataframe')))

    def test_get_data_ids(self):
        self.assertEqual(
            {
                'cube-1-250-250.zarr',
                'cube-5-100-200.zarr',
                'cube.nc',
            },
            set(self.store.get_data_ids())
        )
        self.assertEqual(
            {
                'cube-1-250-250.zarr',
                'cube-5-100-200.zarr',
                'cube.nc',
            },
            set(self.store.get_data_ids('*'))
        )
        self.assertEqual(
            {
                'cube-1-250-250.zarr',
                'cube-5-100-200.zarr',
                'cube.nc',
            },
            set(self.store.get_data_ids('dataset'))
        )
        self.assertEqual(
            set(),
            set(self.store.get_data_ids('dataset[multilevel]'))
        )
        data_ids_list = list(self.store.get_data_ids(include_attrs=["title"]))
        self.assertEqual(3, len(data_ids_list))
        # Note, although we expect "title" to be included,
        # DirectoryStore does not implement it yet.
        self.assertIn(('cube-1-250-250.zarr', {}), data_ids_list)
        self.assertIn(('cube-5-100-200.zarr', {}), data_ids_list)
        self.assertIn(('cube.nc', {}), data_ids_list)
        self.assertEqual(
            {
                'cube-1-250-250.zarr',
                'cube-5-100-200.zarr',
                'cube.nc',
            },
            set(self.store.get_data_ids('dataset'))
        )
        self.assertEqual(
            set(),
            set(self.store.get_data_ids('dataset[multilevel]'))
        )

    def test_has_data(self):
        self.assertTrue(self.store.has_data('cube-1-250-250.zarr'))
        self.assertTrue(self.store.has_data('cube.nc'))
        self.assertFalse(self.store.has_data('cube.levels'))
        self.assertTrue(self.store.has_data('cube-1-250-250.zarr', type_specifier='dataset'))
        self.assertFalse(self.store.has_data('cube-1-250-250.zarr', type_specifier='geodataframe'))

    def test_describe_data(self):
        data_descriptor = self.store.describe_data('cube-1-250-250.zarr')
        self.assertIsInstance(data_descriptor, DatasetDescriptor)
        self.assertEqual('cube-1-250-250.zarr', data_descriptor.data_id)
        self.assertEqual('dataset[cube]', data_descriptor.type_specifier)
        d = data_descriptor.to_dict()
        self.assertIsInstance(d, dict)
        # Assert is JSON-serializable
        json.dumps(d)

        data_descriptor = self.store.describe_data('cube-1-250-250.zarr', type_specifier='dataset')
        self.assertIsInstance(data_descriptor, DatasetDescriptor)
        self.assertEqual('cube-1-250-250.zarr', data_descriptor.data_id)
        self.assertEqual('dataset[cube]', data_descriptor.type_specifier)
        d = data_descriptor.to_dict()
        self.assertIsInstance(d, dict)
        # Assert is JSON-serializable
        json.dumps(d)

        data_descriptor = self.store.describe_data('cube.nc')
        self.assertIsInstance(data_descriptor, DatasetDescriptor)
        self.assertEqual('cube.nc', data_descriptor.data_id)
        self.assertEqual('dataset[cube]', data_descriptor.type_specifier)
        d = data_descriptor.to_dict()
        self.assertIsInstance(d, dict)
        # Assert is JSON-serializable
        json.dumps(d)

        with self.assertRaises(DataStoreError) as cm:
            self.store.describe_data('cube-1-250-250.zarr', type_specifier='geodataframe')
        self.assertEqual(
            'Type specifier "geodataframe" cannot be satisfied by type specifier "dataset" '
            'of data resource "cube-1-250-250.zarr"', f'{cm.exception}')

    def test_search_data_search(self):
        result = list(self.store.search_data())
        self.assertEqual(3, len(result))
        self.assertEqual(result[0].data_id, 'cube-1-250-250.zarr')
        self.assertEqual(result[0].type_specifier, TYPE_SPECIFIER_CUBE)
        self.assertEqual(result[1].data_id, 'cube-5-100-200.zarr')
        self.assertEqual(result[1].type_specifier, TYPE_SPECIFIER_CUBE)
        self.assertEqual(result[2].data_id, 'cube.nc')
        self.assertEqual(result[2].type_specifier, TYPE_SPECIFIER_CUBE)

        result = list(self.store.search_data('*'))
        self.assertEqual(3, len(result))
        self.assertEqual(result[0].data_id, 'cube-1-250-250.zarr')
        self.assertEqual(result[0].type_specifier, TYPE_SPECIFIER_CUBE)
        self.assertEqual(result[1].data_id, 'cube-5-100-200.zarr')
        self.assertEqual(result[1].type_specifier, TYPE_SPECIFIER_CUBE)
        self.assertEqual(result[2].data_id, 'cube.nc')
        self.assertEqual(result[2].type_specifier, TYPE_SPECIFIER_CUBE)

        result = list(self.store.search_data(type_specifier='dataset'))
        self.assertEqual(3, len(result))
        self.assertEqual(result[0].data_id, 'cube-1-250-250.zarr')
        self.assertEqual(result[0].type_specifier, TYPE_SPECIFIER_CUBE)
        self.assertEqual(result[1].data_id, 'cube-5-100-200.zarr')
        self.assertEqual(result[1].type_specifier, TYPE_SPECIFIER_CUBE)
        self.assertEqual(result[2].data_id, 'cube.nc')
        self.assertEqual(result[2].type_specifier, TYPE_SPECIFIER_CUBE)

        result = list(self.store.search_data(type_specifier='geodataframe'))
        self.assertEqual(0, len(result))

        with self.assertRaises(DataStoreError) as cm:
            list(self.store.search_data(type_specifier=TYPE_SPECIFIER_DATASET,
                                        time_range=['2020-03-01', '2020-03-04'],
                                        bbox=[52, 11, 54, 12]))
        self.assertEqual('Unsupported search parameters: time_range, bbox', f'{cm.exception}')

    def test_get_filename_ext(self):
        import xarray as xr
        import geopandas as gpd
        from xcube.core.mldataset import BaseMultiLevelDataset

        dataset = xr.Dataset()
        self.assertEqual('.zarr', self.store._get_filename_ext(dataset))
        frame = gpd.GeoDataFrame()
        self.assertEqual('.geojson', self.store._get_filename_ext(frame))
        mldataset = BaseMultiLevelDataset(base_dataset=dataset)
        self.assertEqual('.levels', self.store._get_filename_ext(mldataset))

        self.assertIsNone(self.store._get_filename_ext(None))
        self.assertIsNone(self.store._get_filename_ext(DataStoreError('A nonsense object')))


class WritableDirectoryDataStoreTest(unittest.TestCase):

    def setUp(self) -> None:
        self._base_dir = os.path.join(os.path.dirname(__file__), 'dir_store_test')
        # make sure self._base_dir is empty
        if os.path.exists(self._base_dir):
            shutil.rmtree(self._base_dir)
        os.mkdir(self._base_dir)
        self._store = new_data_store('directory', base_dir=self._base_dir)
        self.assertIsInstance(self.store, DirectoryDataStore)
        # noinspection PyUnresolvedReferences
        self.assertEqual(self._base_dir, self._store.base_dir)
        # noinspection PyUnresolvedReferences
        self.assertEqual(False, self._store.read_only)

    def tearDown(self) -> None:
        for data_id in self.store.get_data_ids():
            self.store.delete_data(data_id)
        shutil.rmtree(self._base_dir)

    @property
    def store(self) -> DirectoryDataStore:
        # noinspection PyTypeChecker
        return self._store

    def test_write_dataset_only_data(self):
        cube = new_cube()
        cube_id = self.store.write_data(cube)
        self.assertIsNotNone(cube_id)
        self.assertTrue(self.store.has_data(cube_id))
        cube_from_store = self.store.open_data(cube_id)
        self.assertIsNotNone(cube_from_store)

    def test_write_dataset_data_id(self):
        cube = new_cube()
        cube_id = self.store.write_data(cube, data_id='newcube.nc')
        self.assertEquals('newcube.nc', cube_id)
        self.assertTrue(self.store.has_data(cube_id))
        cube_from_store = self.store.open_data(cube_id)
        self.assertIsNotNone(cube_from_store)

    def test_write_dataset_data_id_without_extension(self):
        cube = new_cube()
        cube_id = self.store.write_data(cube, data_id='newcube')
        self.assertEquals('newcube.zarr', cube_id)
        self.assertTrue(self.store.has_data(cube_id))
        cube_from_store = self.store.open_data(cube_id)
        self.assertIsNotNone(cube_from_store)

    def test_write_dataset_invalid_data_id(self):
        cube = new_cube()
        try:
            self.store.write_data(cube, data_id='newcube.shp')
        except DataStoreError as dse:
            self.assertEqual('Data id "newcube.shp" is not suitable for data of type '
                             '"dataset[cube]".',
                             str(dse))

    def test_write_dataset_writer_id(self):
        cube = new_cube()
        cube_id = self.store.write_data(cube, writer_id='dataset:netcdf:posix')
        self.assertTrue(cube_id.endswith('.nc'))
        self.assertTrue(self.store.has_data(cube_id))
        cube_from_store = self.store.open_data(cube_id)
        self.assertIsNotNone(cube_from_store)

    def test_write_dataset_invalid_writer_id(self):
        cube = new_cube()
        try:
            self.store.write_data(cube, writer_id='geodataframe:shapefile:posix')
        except TypeError as te:
            self.assertEqual("data must be an instance of "
                             "(<class 'geopandas.geodataframe.GeoDataFrame'>, "
                             "<class 'pandas.core.frame.DataFrame'>), was "
                             "<class 'xarray.core.dataset.Dataset'>", str(te))

    def test_write_dataset_data_id_and_writer_id(self):
        cube = new_cube()
        cube_id = self.store.write_data(cube,
                                        data_id='newcube.nc',
                                        writer_id='dataset:netcdf:posix')
        self.assertEquals('newcube.nc', cube_id)
        self.assertTrue(self.store.has_data(cube_id))
        cube_from_store = self.store.open_data(cube_id)
        self.assertIsNotNone(cube_from_store)

    def test_write_dataset_invalid_data_id_and_writer_id(self):
        cube = new_cube()
        try:
            self.store.write_data(cube, data_id='newcube.nc', writer_id='dataset:zarr:posix')
        except DataStoreError as dse:
            self.assertEqual('Writer ID "dataset:zarr:posix" seems inappropriate for '
                             'data id "newcube.nc". Try writer id "dataset:netcdf:posix" instead.',
                             str(dse))
