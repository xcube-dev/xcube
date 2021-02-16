import json
import unittest

import requests_mock
import xarray as xr
import yaml

from test.s3test import MOTO_SERVER_ENDPOINT_URL
from test.s3test import S3Test

from xcube.core.gen2.main import main
from xcube.core.dsio import rimraf
from xcube.core.new import new_cube
from xcube.core.store import DataStorePool
from xcube.core.store.stores.memory import MemoryDataStore


class MainTest(unittest.TestCase):
    REQUEST = dict(input_config=dict(store_id='memory',
                                     data_id='S2L2A'),
                   cube_config=dict(variable_names=['B01', 'B02', 'B03'],
                                    crs='WGS84',
                                    bbox=[12.2, 52.1, 13.9, 54.8],
                                    spatial_res=0.05,
                                    time_range=['2018-01-01', None],
                                    time_period='4D'),
                   output_config=dict(store_id='memory',
                                      data_id='CHL'),
                   callback_config=dict(api_uri='https://xcube-gen.test/api/v1/jobs/tomtom/iamajob/callback',
                                        access_token='dfsvdfsv'))

    def setUp(self) -> None:
        with open('_request.json', 'w') as fp:
            json.dump(MainTest.REQUEST, fp)
        with open('_request.yaml', 'w') as fp:
            yaml.dump(MainTest.REQUEST, fp)
        self.saved_cube_memory = MemoryDataStore.replace_global_data_dict(
            {'S2L2A': new_cube(variables=dict(B01=0.1, B02=0.2, B03=0.3))}
        )

    def tearDown(self) -> None:
        rimraf('_request.json')
        rimraf('_request.yaml')
        MemoryDataStore.replace_global_data_dict(self.saved_cube_memory)

    @requests_mock.Mocker()
    def test_json(self, m):
        m.put('https://xcube-gen.test/api/v1/jobs/tomtom/iamajob/callback', json={})
        main('_request.json', verbose=True)
        self.assertIsInstance(MemoryDataStore.get_global_data_dict().get('CHL'),
                              xr.Dataset)

    @requests_mock.Mocker()
    def test_yaml(self, m):
        m.put('https://xcube-gen.test/api/v1/jobs/tomtom/iamajob/callback', json={})
        main('_request.yaml', verbose=True)
        self.assertIsInstance(MemoryDataStore.get_global_data_dict().get('CHL'),
                              xr.Dataset)


class S3MainTest(S3Test):
    BUCKET_NAME = 'xcube-s3-test'
    S3_REQUEST = dict(
        input_config=dict(
            store_id='@my-s3-store',
            data_id='a_b.zarr'),
        cube_config=dict(
            variable_names=['absorbing_aerosol_index'],
            crs='epsg:4326',
            bbox=[0, -20, 20, 20],
            spatial_res=1.0,
            time_range=['2005-02-01', '2005-02-28'],
            time_period='1M'
        ),
        output_config=dict(
            store_id='memory',
            data_id='A_B'),
        callback_config=dict(
            api_uri='https://xcube-gen.test/api/v1/jobs/tomtom/iamajob/callback',
            access_token='dfsvdfsv'))
    STORE_CONFIGS = {
        "my-s3-store": {
            "store_id": "s3",
            "store_params": {
                "aws_access_key_id": "test_fake_id",
                "aws_secret_access_key": "test_fake_secret",
                "bucket_name": BUCKET_NAME,
                "endpoint_url": MOTO_SERVER_ENDPOINT_URL
            }
        }
    }

    def setUp(self):
        super().setUp()
        pool = DataStorePool.from_dict(S3MainTest.STORE_CONFIGS)
        s3_store = pool.get_store("my-s3-store")
        cube = new_cube(variables=dict(a=4.1, b=7.4))
        s3_store.s3.mkdir(S3MainTest.BUCKET_NAME)
        s3_store.write_data(cube, 'a_b.zarr')
        with open('_s3_request.json', 'w') as fp:
            json.dump(S3MainTest.S3_REQUEST, fp)
        with open('_s3_store_configs.json', 'w') as fp:
            json.dump(S3MainTest.STORE_CONFIGS, fp)
        self.saved_cube_memory = MemoryDataStore.get_global_data_dict().copy()

    def tearDown(self) -> None:
        rimraf('_s3_request.json')
        rimraf('_s3_store_configs.json')
        MemoryDataStore.replace_global_data_dict(self.saved_cube_memory)

    @requests_mock.Mocker()
    def test_json(self, m):
        m.put('https://xcube-gen.test/api/v1/jobs/tomtom/iamajob/callback', json={})
        main('_s3_request.json', store_configs_path='_s3_store_configs.json', verbose=True)
        self.assertIsInstance(MemoryDataStore.get_global_data_dict().get('A_B'),
                              xr.Dataset)
