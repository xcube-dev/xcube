import json
import unittest

import xarray as xr
import yaml

from xcube.cli._gen2.main import main
from xcube.core.dsio import rimraf
from xcube.core.new import new_cube
from xcube.core.store.stores.memory import MemoryCubeStore


class MainTest(unittest.TestCase):
    REQUEST = dict(input_configs=[dict(cube_store_id='mem',
                                       cube_id='S2L2A',
                                       variable_names=['B01', 'B02', 'B03'])],
                   cube_config=dict(spatial_crs='WGS84',
                                    spatial_coverage=[12.2, 52.1, 13.9, 54.8],
                                    spatial_resolution=0.05,
                                    temporal_coverage=['2018-01-01', None],
                                    temporal_resolution='4D'),
                   output_config=dict(cube_store_id='mem',
                                      cube_id='CHL'))

    def setUp(self) -> None:
        with open('_request.json', 'w') as fp:
            json.dump(MainTest.REQUEST, fp)
        with open('_request.yaml', 'w') as fp:
            yaml.dump(MainTest.REQUEST, fp)
        self.saved_cube_memory = MemoryCubeStore.replace_global_cube_memory(
            {'S2L2A': new_cube(variables=dict(B01=0.1, B02=0.2, B03=0.3))}
        )

    def tearDown(self) -> None:
        rimraf('_request.json')
        rimraf('_request.yaml')
        MemoryCubeStore.replace_global_cube_memory(self.saved_cube_memory)

    def test_json(self):
        main('_request.json')
        self.assertIsInstance(MemoryCubeStore.get_global_data_storage().get('CHL'),
                              xr.Dataset)

    def test_yaml(self):
        main('_request.yaml')
        self.assertIsInstance(MemoryCubeStore.get_global_data_storage().get('CHL'),
                              xr.Dataset)
