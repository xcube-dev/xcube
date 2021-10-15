import json
import os
import unittest

from xcube.core.gen2.config import CubeConfig
from xcube.core.gen2.config import InputConfig
from xcube.core.gen2.config import OutputConfig
from xcube.core.gen2.request import CubeGeneratorRequest


class CubeGeneratorRequestTest(unittest.TestCase):
    def test_normalize(self):
        request_dict = dict(
            input_config=dict(data_id='S2L2A', store_id='memory'),
            cube_config=dict(variable_names=['B01', 'B02'],
                             crs='WGS84',
                             bbox=[12.2, 52.1, 13.9, 54.8],
                             spatial_res=0.05,
                             time_range=['2018-01-01', None],
                             time_period='4D'),
            output_config=dict(store_id='memory',
                               replace=False,
                               data_id='CHL')
        )
        file_path = '_test-request.json'
        with open(file_path, 'w') as fp:
            json.dump(request_dict, fp)
        try:
            request_1 = CubeGeneratorRequest.normalize(file_path)
            self.assertIsInstance(request_1, CubeGeneratorRequest)
            request_2 = CubeGeneratorRequest.normalize(request_1)
            self.assertIs(request_2, request_1)
            request_3 = CubeGeneratorRequest.normalize(request_2.to_dict())
            self.assertIsInstance(request_3, CubeGeneratorRequest)
            self.assertEqual(request_dict, request_3.to_dict())
        finally:
            os.remove(file_path)

        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            CubeGeneratorRequest.normalize(42)

    def test_from_dict(self):
        request_dict = dict(
            input_configs=[dict(store_id='memory',
                                data_id='S2L2A')],
            cube_config=dict(variable_names=['B01', 'B02'],
                             crs='WGS84',
                             bbox=[12.2, 52.1, 13.9, 54.8],
                             spatial_res=0.05,
                             time_range=['2018-01-01', None],
                             time_period='4D'),
            output_config=dict(store_id='memory',
                               data_id='CHL')
        )
        gen_config = CubeGeneratorRequest.from_dict(request_dict)
        self.assertIsInstance(gen_config, CubeGeneratorRequest)
        self.assertEqual(1, len(gen_config.input_configs))
        self.assertIsInstance(gen_config.input_configs[0], InputConfig)
        self.assertEqual('memory', gen_config.input_configs[0].store_id)
        self.assertEqual('S2L2A', gen_config.input_configs[0].data_id)
        self.assertIsInstance(gen_config.output_config, OutputConfig)
        self.assertEqual('memory', gen_config.output_config.store_id)
        self.assertEqual('CHL', gen_config.output_config.data_id)
        self.assertIsInstance(gen_config.cube_config, CubeConfig)
        self.assertEqual(('B01', 'B02'), gen_config.cube_config.variable_names)
        self.assertEqual('WGS84', gen_config.cube_config.crs)
        self.assertEqual((12.2, 52.1, 13.9, 54.8), gen_config.cube_config.bbox)
        self.assertEqual(0.05, gen_config.cube_config.spatial_res)
        self.assertEqual(('2018-01-01', None), gen_config.cube_config.time_range)
        self.assertEqual('4D', gen_config.cube_config.time_period)

    def test_to_dict(self):
        expected_dict = dict(
            input_config=dict(store_id='memory',
                              data_id='S2L2A'),
            cube_config=dict(variable_names=['B01', 'B02'],
                             crs='WGS84',
                             bbox=[12.2, 52.1, 13.9, 54.8],
                             spatial_res=0.05,
                             time_range=['2018-01-01', None],
                             time_period='4D'),
            output_config=dict(store_id='memory',
                               replace=False,
                               data_id='CHL')
        )
        request = CubeGeneratorRequest.from_dict(expected_dict)
        actual_dict = request.to_dict()
        self.assertEqual(expected_dict, actual_dict)
        # JSON-serialisation smoke test
        json.dumps(actual_dict, indent=2)

        # No cube config is ok too
        expected_dict = dict(
            input_config=dict(store_id='memory',
                              data_id='S2L2A'),
            output_config=dict(store_id='memory',
                               replace=False,
                               data_id='CHL')
        )
        request = CubeGeneratorRequest.from_dict(expected_dict)
        actual_dict = request.to_dict()
        self.assertEqual(expected_dict, actual_dict)
        # JSON-serialisation smoke test
        json.dumps(actual_dict, indent=2)
