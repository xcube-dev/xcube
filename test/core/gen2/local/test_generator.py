import copy
import json
import unittest
from typing import Dict, Any

import requests_mock
import xarray as xr
import yaml

from xcube.core.dsio import rimraf
from xcube.core.gen2.local.generator import LocalCubeGenerator
from xcube.core.gen2.response import CubeGeneratorResult
from xcube.core.gen2.response import CubeInfo
from xcube.core.gen2.response import CubeInfoResult
from xcube.core.gen2.response import CubeReference
from xcube.core.new import new_cube
from xcube.core.store import DatasetDescriptor
from xcube.core.store import MutableDataStore
from xcube.core.store import new_data_store

CALLBACK_MOCK_URL = 'https://xcube-gen.test/api/v1/jobs/tomtom/iamajob/callback'


class LocalCubeGeneratorTest(unittest.TestCase):
    REQUEST: Dict[str, Any] = dict(
        input_config=dict(
            store_id='memory',
            data_id='S2L2A.zarr'
        ),
        cube_config=dict(
            variable_names=['B01', 'B02', 'B03'],
            crs='WGS84',
            bbox=[12.2, 52.1, 13.9, 54.8],
            spatial_res=0.05,
            time_range=['2010-01-01', None],
            time_period='4D',
            chunks=dict(time=None, lat=90, lon=90),
            metadata=dict(title='A S2L2A subset'),
            variable_metadata=dict(
                B01=dict(long_name='Band 1'),
                B02=dict(long_name='Band 2'),
                B03=dict(long_name='Band 3'),
            ),
        ),
        output_config=dict(
            store_id='memory',
            data_id='CHL.zarr',
            replace=True,
        ),
        callback_config=dict(
            api_uri=CALLBACK_MOCK_URL,
            access_token='dfsvdfsv'
        )
    )

    def setUp(self) -> None:
        self.data_store: MutableDataStore = new_data_store('memory')
        with open('_request.json', 'w') as fp:
            json.dump(self.REQUEST, fp)
        with open('_request.yaml', 'w') as fp:
            yaml.dump(self.REQUEST, fp)
        self.data_store.write_data(
            new_cube(variables=dict(B01=0.1, B02=0.2, B03=0.3)),
            'S2L2A.zarr',
            replace=True,
        )

    def tearDown(self) -> None:
        rimraf('_request.json')
        rimraf('_request.yaml')
        self.data_store.delete_data('S2L2A.zarr')
        if self.data_store.has_data('CHL.zarr'):
            self.data_store.delete_data('CHL.zarr')

    @requests_mock.Mocker()
    def test_generate_cube_from_dict(self, m):
        m.put(CALLBACK_MOCK_URL, json={})
        self.assertGeneratedCubeOk(self.REQUEST)

    @requests_mock.Mocker()
    def test_generate_cube_from_json(self, m):
        m.put(CALLBACK_MOCK_URL, json={})
        self.assertGeneratedCubeOk('_request.json')

    @requests_mock.Mocker()
    def test_generate_cube_from_yaml(self, m):
        m.put(CALLBACK_MOCK_URL, json={})
        self.assertGeneratedCubeOk('_request.yaml')

    def assertGeneratedCubeOk(self, request):
        generator = LocalCubeGenerator(verbosity=1)
        result = generator.generate_cube(request)
        self.assertIsInstance(result, CubeGeneratorResult)
        self.assertEqual('ok', result.status)
        self.assertEqual(201, result.status_code)
        cube_reference = result.result
        self.assertIsInstance(cube_reference, CubeReference)
        self.assertEqual('CHL.zarr', cube_reference.data_id)
        dataset = self.data_store.open_data(cube_reference.data_id)
        self.assertIsInstance(dataset, xr.Dataset)
        self.assertIn('B01', dataset)
        self.assertIn('B02', dataset)
        self.assertIn('B03', dataset)
        self.assertEqual('A S2L2A subset', dataset.attrs.get('title'))
        self.assertEqual('Band 1', dataset.B01.attrs.get('long_name'))
        self.assertEqual('Band 2', dataset.B02.attrs.get('long_name'))
        self.assertEqual('Band 3', dataset.B03.attrs.get('long_name'))

    @requests_mock.Mocker()
    def test_generate_cube_from_yaml_empty(self, m):
        m.put(CALLBACK_MOCK_URL, json={})
        request = copy.deepcopy(self.REQUEST)
        request['cube_config']['time_range'] = ['2019-01-01', '2020-01-01']

        generator = LocalCubeGenerator()
        result = generator.generate_cube(request)
        self.assertEqual('warning', result.status)
        self.assertEqual(422, result.status_code)
        self.assertEqual(None, result.result)
        self.assertIsInstance(result.message, str)
        self.assertIn('An empty cube has been generated', result.message)
        self.assertEqual(None, generator.generated_data_id)
        self.assertEqual(None, generator.generated_cube, xr.Dataset)
        self.assertEqual(None, generator.generated_gm)

    @requests_mock.Mocker()
    def test_generate_cube_with_internal_error(self, m):
        m.put(CALLBACK_MOCK_URL, json={})
        request = self.REQUEST.copy()
        request['cube_config']['metadata'] = {
            'inverse_fine_structure_constant': 138
        }

        generator = LocalCubeGenerator()
        with self.assertRaises(ValueError) as cm:
            generator.generate_cube(request)
        self.assertEqual(('inverse_fine_structure_constant must be 137'
                          ' or running in wrong universe',),
                         cm.exception.args)

    @requests_mock.Mocker()
    def test_get_cube_info_from_dict(self, m):
        m.put(CALLBACK_MOCK_URL, json={})
        result = LocalCubeGenerator(verbosity=1).get_cube_info(self.REQUEST)
        self.assertIsInstance(result, CubeInfoResult)
        self.assertEqual('ok', result.status)
        self.assertEqual(200, result.status_code)
        cube_info = result.result
        self.assertIsInstance(cube_info, CubeInfo)
        self.assertIsInstance(cube_info.dataset_descriptor, DatasetDescriptor)
        self.assertIsInstance(cube_info.size_estimation, dict)
        # JSON-serialisation smoke test
        cube_info_dict = result.to_dict()
        json.dumps(cube_info_dict, indent=2)

# class CubeGeneratorS3Input(S3Test):
#     BUCKET_NAME = 'xcube-s3-test'
#     S3_REQUEST = dict(
#         input_config=dict(
#             store_id='@my-s3-store',
#             data_id='a_b.zarr'),
#         cube_config=dict(
#             variable_names=['absorbing_aerosol_index'],
#             crs='epsg:4326',
#             bbox=[0, -20, 20, 20],
#             spatial_res=1.0,
#             time_range=['2005-02-01', '2005-02-28'],
#             time_period='1M'
#         ),
#         output_config=dict(
#             store_id='memory',
#             data_id='A_B'),
#         callback_config=dict(
#             api_uri='https://xcube-gen.test/api/v1/jobs/tomtom/iamajob/callback',
#             access_token='dfsvdfsv'))
#     STORE_CONFIGS = {
#         "my-s3-store": {
#             "store_id": "s3",
#             "store_params": {
#                 "aws_access_key_id": "test_fake_id",
#                 "aws_secret_access_key": "test_fake_secret",
#                 "bucket_name": BUCKET_NAME,
#                 "endpoint_url": MOTO_SERVER_ENDPOINT_URL
#             }
#         }
#     }
#
#     def setUp(self):
#         super().setUp()
#         pool = DataStorePool.from_dict(self.STORE_CONFIGS)
#         s3_store = pool.get_store("my-s3-store")
#         cube = new_cube(variables=dict(a=4.1, b=7.4))
#         s3_store.s3.mkdir(self.BUCKET_NAME)
#         s3_store.write_data(cube, 'a_b.zarr')
#         import time
#         time.sleep(5)
#         self.assertTrue(s3_store.has_data('a_b.zarr'))
#         with open('_s3_request.json', 'w') as fp:
#             json.dump(self.S3_REQUEST, fp)
#         with open('_s3_store_configs.json', 'w') as fp:
#             json.dump(self.STORE_CONFIGS, fp)
#
#     def tearDown(self) -> None:
#         rimraf('_s3_request.json')
#         rimraf('_s3_store_configs.json')
#
#     @requests_mock.Mocker()
#     def test_json(self, m):
#         m.put('https://xcube-gen.test/api/v1/jobs/tomtom/iamajob/callback', json={})
#         CubeGenerator.from_file('_s3_request.json', store_configs_path='_s3_store_configs.json', verbose=True).run()
#         self.assertIsInstance(MemoryDataStore.get_global_data_dict().get('A_B'),
#                               xr.Dataset)
