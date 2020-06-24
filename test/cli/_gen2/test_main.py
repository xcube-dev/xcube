import json
import unittest

import requests_mock
import xarray as xr
import yaml

from xcube.cli._gen2.main import main
from xcube.core.dsio import rimraf
from xcube.core.new import new_cube
from xcube.core.store.stores.memory import MemoryDataStore


class MainTest(unittest.TestCase):
    REQUEST = dict(input_configs=[dict(store_id='memory',
                                       data_id='S2L2A',
                                       variable_names=['B01', 'B02', 'B03'])],
                   cube_config=dict(crs='WGS84',
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
        main('_request.json')
        self.assertIsInstance(MemoryDataStore.get_global_data_dict().get('CHL'),
                              xr.Dataset)

    @requests_mock.Mocker()
    def test_yaml(self, m):
        m.put('https://xcube-gen.test/api/v1/jobs/tomtom/iamajob/callback', json={})
        main('_request.yaml')
        self.assertIsInstance(MemoryDataStore.get_global_data_dict().get('CHL'),
                              xr.Dataset)
