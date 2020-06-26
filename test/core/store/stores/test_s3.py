import unittest

import boto3
import moto
import xarray as xr

from xcube.core.new import new_cube
from xcube.core.store import DataStoreError
from xcube.core.store import new_data_store
from xcube.core.store.stores.s3 import S3DataStore

BUCKET_NAME = 'xcube-test'


class S3DataStoreTest(unittest.TestCase):

    def setUp(self) -> None:
        self._store = new_data_store('s3',
                                     aws_access_key_id='test_fake_id',
                                     aws_secret_access_key='test_fake_secret',
                                     bucket_name=BUCKET_NAME)
        self.assertIsInstance(self.store, S3DataStore)

    @property
    def store(self) -> S3DataStore:
        # noinspection PyTypeChecker
        return self._store

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
        self.assertEqual(('dataset',), self.store.get_type_ids())

    def test_get_data_opener_ids(self):
        self.assertEqual(('dataset:zarr:s3',), self.store.get_data_opener_ids())

    def test_get_data_writer_ids(self):
        self.assertEqual(('dataset:zarr:s3',), self.store.get_data_writer_ids())

    def test_write_and_read_and_delete(self):
        with moto.mock_s3():
            s3_conn = boto3.client('s3')
            s3_conn.create_bucket(Bucket=BUCKET_NAME, ACL='public-read')

            dataset_1 = new_cube(variables=dict(a=4.1, b=7.4))
            dataset_2 = new_cube(variables=dict(c=5.2, d=8.5))
            dataset_3 = new_cube(variables=dict(e=6.3, f=9.6))

            # Write 3 cubes
            self.store.write_data(dataset_1, data_id='cube-1.zarr')
            self.store.write_data(dataset_2, data_id='cube-2.zarr')
            self.store.write_data(dataset_3, data_id='cube-3.zarr')

            self.assertEqual({'cube-1.zarr',
                              'cube-2.zarr',
                              'cube-3.zarr'},
                             set(self.store.get_data_ids()))

            self.assertTrue(self.store.has_data('cube-1.zarr'))
            self.assertTrue(self.store.has_data('cube-2.zarr'))
            self.assertTrue(self.store.has_data('cube-3.zarr'))

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
            self.assertEqual({'cube-2.zarr', 'cube-3.zarr'},
                             set(self.store.get_data_ids()))
            self.assertFalse(self.store.has_data('cube-1.zarr'))

            # Now it should be save to also write with replace=False
            self.store.write_data(dataset_1, data_id='cube-1.zarr', replace=False)
