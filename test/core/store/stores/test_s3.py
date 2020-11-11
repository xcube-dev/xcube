import os
import unittest

import s3fs
import xarray as xr
import numpy as np

from test.s3test import S3Test, MOTO_SERVER_ENDPOINT_URL
from xcube.cli.prune import _prune
from xcube.core.dsio import rimraf, write_cube
from xcube.core.new import new_cube
from xcube.core.store import DataStoreError
from xcube.core.store import new_data_store
from xcube.core.store.stores.s3 import S3DataStore

BUCKET_NAME = 'xcube-test'


def get_pruned_cube(path):
    _prune(input_path=path,
           dry_run=False,
           monitor=print)
    dataset_p = xr.open_zarr(path)
    return dataset_p


class S3DataStoreTest(S3Test):
    TEST_CUBE = "test.zarr"

    def setUp(self) -> None:
        super().setUp()
        self._store = new_data_store('s3',
                                     aws_access_key_id='test_fake_id',
                                     aws_secret_access_key='test_fake_secret',
                                     bucket_name=BUCKET_NAME,
                                     endpoint_url=MOTO_SERVER_ENDPOINT_URL)
        self.assertIsInstance(self.store, S3DataStore)
        rimraf(self.TEST_CUBE)
        cube = new_cube(time_periods=3,
                        variables=dict(precipitation=np.nan,
                                       temperature=np.nan)).chunk(dict(time=1, lat=90, lon=90))

        write_cube(cube, self.TEST_CUBE, "zarr", cube_asserted=True)

    def tearDown(self) -> None:
        rimraf(self.TEST_CUBE)

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
        self.assertEqual(('dataset:zarr:s3',), self.store.get_data_opener_ids(type_specifier='dataset'))
        self.assertEqual(('dataset:zarr:s3',), self.store.get_data_opener_ids(type_specifier='*'))
        with self.assertRaises(ValueError) as cm:
            self.store.get_data_opener_ids(type_specifier='dataset[cube]')
        self.assertEqual("type_specifier must be one of ('dataset',)", f'{cm.exception}')

    def test_get_data_writer_ids(self):
        self.assertEqual(('dataset:zarr:s3',), self.store.get_data_writer_ids())
        self.assertEqual(('dataset:zarr:s3',), self.store.get_data_writer_ids(type_specifier='dataset'))
        self.assertEqual(('dataset:zarr:s3',), self.store.get_data_writer_ids(type_specifier='*'))
        with self.assertRaises(ValueError) as cm:
            self.store.get_data_writer_ids(type_specifier='dataset[cube]')
        self.assertEqual("type_specifier must be one of ('dataset',)", f'{cm.exception}')

    @unittest.skip('Currently fails on travis but not locally, execute on demand only')
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

    def test_get_pruned_datacube(self):
        self.store.s3.mkdir(BUCKET_NAME)
        cube = new_cube(time_periods=3,
                        variables=dict(precipitation=np.nan,
                                       temperature=np.nan)).chunk(dict(time=1, lat=90, lon=90))

        write_cube(cube, self.TEST_CUBE, cube_asserted=True)
        dataset_p = get_pruned_cube(f'{os.getcwd()}/{self.TEST_CUBE}')
        self.store.write_data(dataset_p, data_id='cube-pruned.zarr')

        self.assertTrue(self.store.has_data('cube-pruned.zarr'))
        self.assertIn(('cube-pruned.zarr', None), set(self.store.get_data_ids()))
        # Open the written cube
        opened_dataset_1 = self.store.open_data('cube-pruned.zarr')
        self.assertIsInstance(opened_dataset_1, xr.Dataset)
        self.assertEqual(set(dataset_p.data_vars), set(opened_dataset_1.data_vars))
        self.assertEqual(dataset_p.precipitation.values.all(), opened_dataset_1.precipitation.values.all())
