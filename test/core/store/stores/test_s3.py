import json
import unittest

import s3fs
import xarray as xr

from test.s3test import S3Test, MOTO_SERVER_ENDPOINT_URL
from xcube.core.new import new_cube
from xcube.core.store import DataStoreError
from xcube.core.store import DatasetDescriptor
from xcube.core.store import TYPE_SPECIFIER_CUBE
from xcube.core.store import TYPE_SPECIFIER_DATASET
from xcube.core.store import TYPE_SPECIFIER_MULTILEVEL_DATASET
from xcube.core.store import new_data_store
from xcube.core.store.stores.s3 import S3DataStore
from xcube.util.jsonschema import JsonObjectSchema

BUCKET_NAME = 'xcube-test'


class S3DataStoreTest(S3Test):

    def setUp(self) -> None:
        super().setUp()
        self._store = new_data_store('s3',
                                     aws_access_key_id='test_fake_id',
                                     aws_secret_access_key='test_fake_secret',
                                     bucket_name=BUCKET_NAME,
                                     endpoint_url=MOTO_SERVER_ENDPOINT_URL)
        self.assertIsInstance(self.store, S3DataStore)

    @property
    def store(self) -> S3DataStore:
        # noinspection PyTypeChecker
        return self._store

    def test_props(self):
        self.assertIsInstance(self.store.s3, s3fs.S3FileSystem)
        self.assertEqual(BUCKET_NAME, self.store.bucket_name)

    def test_get_data_store_params_schema(self):
        schema = self.store.get_data_store_params_schema()
        self.assertEqual(
            {'anon',
             'aws_access_key_id',
             'aws_secret_access_key',
             'aws_session_token',
             'endpoint_url',
             'profile_name',
             'bucket_name',
             'region_name'},
            set(schema.properties.keys())
        )
        self.assertEqual({'bucket_name'}, schema.required)

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

    # TODO (forman): Fixme! Currently get boto3 errors when running out-commented test
    # def test_search_data(self):
    #     result = list(self.store.search_data(type_specifier=TYPE_SPECIFIER_CUBE))
    #     self.assertTrue(len(result) > 0)
    #
    #     result = list(self.store.search_data(type_specifier=TYPE_SPECIFIER_DATASET))
    #     self.assertTrue(len(result) > 0)
    #
    #     with self.assertRaises(DataStoreError) as cm:
    #         list(self.store.search_data(type_specifier=TYPE_SPECIFIER_DATASET,
    #                                     time_range=['2020-03-01', '2020-03-04'],
    #                                     bbox=[52, 11, 54, 12]))
    #     self.assertEqual('Unsupported search parameters: time_range, bbox', f'{cm.exception}')

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
        self.assertEqual(('dataset',), self.store.get_type_specifiers())

    def test_get_data_opener_ids(self):
        self.assertEqual(('dataset:zarr:s3',), self.store.get_data_opener_ids())
        self.assertEqual(('dataset:zarr:s3',),
                         self.store.get_data_opener_ids(type_specifier='dataset'))
        self.assertEqual(('dataset:zarr:s3',),
                         self.store.get_data_opener_ids(type_specifier='*'))
        with self.assertRaises(ValueError) as cm:
            self.store.get_data_opener_ids(type_specifier='dataset[cube]')
        self.assertEqual("type_specifier must be one of ('dataset',)", f'{cm.exception}')

    def test_get_data_writer_ids(self):
        self.assertEqual(('dataset:zarr:s3',), self.store.get_data_writer_ids())
        self.assertEqual(('dataset:zarr:s3',),
                         self.store.get_data_writer_ids(type_specifier='dataset'))
        self.assertEqual(('dataset:zarr:s3',), self.store.get_data_writer_ids(type_specifier='*'))
        with self.assertRaises(ValueError) as cm:
            self.store.get_data_writer_ids(type_specifier='dataset[cube]')
        self.assertEqual("type_specifier must be one of ('dataset',)", f'{cm.exception}')

    def test_data_registration(self):
        self.store.s3.mkdir(BUCKET_NAME)
        dataset = new_cube(variables=dict(a=4.1, b=7.4))
        self.store.register_data(data_id='cube', data=dataset)
        self.assertTrue(self.store.has_data(data_id='cube'))
        self.assertTrue(self.store.has_data(data_id='cube',
                                            type_specifier=TYPE_SPECIFIER_DATASET))
        self.assertTrue(self.store.has_data(data_id='cube',
                                            type_specifier=TYPE_SPECIFIER_CUBE))
        self.assertFalse(self.store.has_data(data_id='cube',
                                             type_specifier=TYPE_SPECIFIER_MULTILEVEL_DATASET))
        self.store.deregister_data(data_id='cube')
        self.assertFalse(self.store.has_data(data_id='cube'))

    def test_write_and_describe_data_from_registry(self):
        self.store.s3.mkdir(BUCKET_NAME)
        dataset_1 = new_cube(variables=dict(a=4.1, b=7.4))
        self.store.write_data(dataset_1, data_id='cube-1.zarr')

        data_descriptor = self.store.describe_data('cube-1.zarr')
        self.assertIsInstance(data_descriptor, DatasetDescriptor)
        self.assertEqual('cube-1.zarr', data_descriptor.data_id)
        self.assertEqual(TYPE_SPECIFIER_CUBE, data_descriptor.type_specifier)
        self.assertEqual((-180.0, -90.0, 180.0, 90.0), data_descriptor.bbox)
        self.assertDictEqual(dict(bnds=2, lat=180, lon=360, time=5), data_descriptor.dims)
        self.assertEqual(('2010-01-01', '2010-01-06'),
                         data_descriptor.time_range)
        self.assertEqual({'a', 'b'}, set(data_descriptor.data_vars.keys()))

        d = data_descriptor.to_dict()
        self.assertIsInstance(d, dict)
        # Assert is JSON-serializable
        json.dumps(d)

    def test_write_and_get_type_specifiers_for_data(self):
        self.store.s3.mkdir(BUCKET_NAME)
        dataset_1 = new_cube(variables=dict(a=4.1, b=7.4))
        self.store.write_data(dataset_1, data_id='cube-1.zarr')

        type_specifiers = self.store.get_type_specifiers_for_data('cube-1.zarr')
        self.assertEqual(1, len(type_specifiers))
        self.assertEqual(('dataset',), type_specifiers)
        self.assertIsInstance(type_specifiers[0], str)
        from xcube.core.store import TypeSpecifier
        TypeSpecifier.parse(type_specifiers[0])

    @unittest.skip('Currently fails on appveyor but not locally, execute on demand only')
    def test_write_and_has_data(self):
        self.assertFalse(self.store.has_data('cube-1.zarr'))

        self.store.s3.mkdir(BUCKET_NAME)
        dataset_1 = new_cube(variables=dict(a=4.1, b=7.4))
        self.store.write_data(dataset_1, data_id='cube-1.zarr')

        self.assertTrue(self.store.has_data('cube-1.zarr'))
        self.assertTrue(self.store.has_data('cube-1.zarr', type_specifier='dataset'))
        self.assertFalse(self.store.has_data('cube-1.zarr', type_specifier='geodataframe'))
        self.assertFalse(self.store.has_data('cube-2.zarr'))

        d = data_descriptor.to_dict()
        self.assertIsInstance(d, dict)
        # Assert is JSON-serializable
        json.dumps(d)

    @unittest.skip('Currently fails on appveyor but not locally, execute on demand only')
    def test_write_and_describe_data_from_zarr_describer(self):
        self.store.s3.mkdir(BUCKET_NAME)
        dataset_1 = new_cube(variables=dict(a=4.1, b=7.4))
        self.store.write_data(dataset_1, data_id='cube-1.zarr')
        self.store.deregister_data('cube-1.zarr')

        data_descriptor = self.store.describe_data('cube-1.zarr')
        self.assertIsInstance(data_descriptor, DatasetDescriptor)
        self.assertEqual('cube-1.zarr', data_descriptor.data_id)
        self.assertEqual(TYPE_SPECIFIER_DATASET, data_descriptor.type_specifier)
        self.assertEqual((-90.0, -180.0, 90.0, 180.0), data_descriptor.bbox)
        self.assertDictEqual(dict(bnds=2, lat=180, lon=360, time=5), data_descriptor.dims)
        self.assertEqual(('2010-01-01T00:00:00', '2010-01-06T00:00:00'),
                         data_descriptor.time_range)
        self.assertEqual({'a', 'b'}, set(data_descriptor.data_vars.keys()))

    @unittest.skip('Currently fails on appveyor but not locally, execute on demand only')
    def test_write_and_read_and_delete(self):
        self.store.s3.mkdir(BUCKET_NAME)

        dataset_1 = new_cube(variables=dict(a=4.1, b=7.4))
        dataset_2 = new_cube(variables=dict(c=5.2, d=8.5))
        dataset_3 = new_cube(variables=dict(e=6.3, f=9.6))

        # Write 3 cubes
        self.store.write_data(dataset_1, data_id='cube-1.zarr')
        self.store.write_data(dataset_2, data_id='cube-2.zarr')
        self.store.write_data(dataset_3, data_id='cube-3.zarr')

        self.assertTrue(self.store.has_data('cube-1.zarr'))
        self.assertTrue(self.store.has_data('cube-2.zarr'))
        self.assertTrue(self.store.has_data('cube-3.zarr'))

        self.assertIn(('cube-1.zarr', None), set(self.store.get_data_ids()))
        self.assertIn(('cube-2.zarr', None), set(self.store.get_data_ids()))
        self.assertIn(('cube-3.zarr', None), set(self.store.get_data_ids()))
        self.assertEqual(3, len(set(self.store.get_data_ids())))

        # Open the 3 written cubes
        opened_dataset_1 = self.store.open_data('cube-1.zarr')
        opened_dataset_2 = self.store.open_data('cube-2.zarr')
        opened_dataset_3 = self.store.open_data('cube-3.zarr')

        self.assertIsInstance(opened_dataset_1, xr.Dataset)
        self.assertIsInstance(opened_dataset_2, xr.Dataset)
        self.assertIsInstance(opened_dataset_3, xr.Dataset)

        self.assertEqual(set(dataset_1.data_vars), set(opened_dataset_1.data_vars))
        self.assertEqual(set(dataset_2.data_vars), set(opened_dataset_2.data_vars))
        self.assertEqual(set(dataset_3.data_vars), set(opened_dataset_3.data_vars))

        # Try overwriting existing cube 1
        dataset_4 = new_cube(variables=dict(g=7.4, h=10.7))
        with self.assertRaises(DataStoreError) as cm:
            self.store.write_data(dataset_4, data_id='cube-1.zarr')
        self.assertEqual("path '' contains a group", f'{cm.exception}')
        # replace=True should do the trick
        self.store.write_data(dataset_4, data_id='cube-1.zarr', replace=True)
        opened_dataset_4 = self.store.open_data('cube-1.zarr')
        self.assertEqual(set(dataset_4.data_vars), set(opened_dataset_4.data_vars))

        # Try deleting cube 1
        self.store.delete_data('cube-1.zarr')
        self.assertEqual({('cube-2.zarr', None), ('cube-3.zarr', None)},
                         set(self.store.get_data_ids()))
        self.assertFalse(self.store.has_data('cube-1.zarr'))

        # Now it should be save to also write with replace=False
        self.store.write_data(dataset_1, data_id='cube-1.zarr', replace=False)
