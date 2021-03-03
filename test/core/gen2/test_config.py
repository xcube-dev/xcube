import json
import unittest

from xcube.core.gen2.config import CallbackConfig
from xcube.core.gen2.config import CubeConfig
from xcube.core.gen2.config import InputConfig
from xcube.core.gen2.config import OutputConfig


class InputConfigTest(unittest.TestCase):

    def test_from_dict(self):
        json_instance = dict(store_id='sentinelhub', data_id='S2L2A')
        input_config = InputConfig.from_dict(json_instance)
        self.assertIsInstance(input_config, InputConfig)
        self.assertEqual('sentinelhub', input_config.store_id)
        self.assertEqual('S2L2A', input_config.data_id)
        self.assertEqual(None, input_config.store_params)
        self.assertEqual(None, input_config.open_params)

    def test_to_dict(self):
        expected_dict = dict(store_id='sentinelhub', data_id='S2L2A')
        input_config = InputConfig(**expected_dict)
        actual_dict = input_config.to_dict()
        self.assertEqual(expected_dict, actual_dict)
        # smoke test JSON serialisation
        json.dumps(actual_dict, indent=2)


class OutputConfigTest(unittest.TestCase):

    def test_from_dict(self):
        json_instance = dict(store_id='s3', data_id='CHL.zarr')
        output_config = OutputConfig.get_schema().from_instance(json_instance)
        self.assertIsInstance(output_config, OutputConfig)
        self.assertEqual('s3', output_config.store_id)
        self.assertEqual('CHL.zarr', output_config.data_id)
        self.assertEqual(None, output_config.store_params)
        self.assertEqual(None, output_config.write_params)

    def test_to_dict(self):
        expected_dict = dict(store_id='s3', replace=False, data_id='CHL.zarr')
        output_config = OutputConfig(**expected_dict)
        actual_dict = output_config.to_dict()
        self.assertEqual(expected_dict, actual_dict)
        # smoke test JSON serialisation
        json.dumps(actual_dict, indent=2)


class CubeConfigTest(unittest.TestCase):

    def test_from_dict(self):
        json_instance = dict(variable_names=['B03', 'B04'],
                             crs='WGS84',
                             bbox=[12.2, 52.1, 13.9, 54.8],
                             spatial_res=0.05,
                             time_range=['2018-01-01', None],
                             time_period='4D')
        cube_config = CubeConfig.from_dict(json_instance)
        self.assertIsInstance(cube_config, CubeConfig)
        self.assertEqual(('B03', 'B04'), cube_config.variable_names)
        self.assertEqual('WGS84', cube_config.crs)
        self.assertEqual((12.2, 52.1, 13.9, 54.8), cube_config.bbox)
        self.assertEqual(0.05, cube_config.spatial_res)
        self.assertEqual(('2018-01-01', None), cube_config.time_range)
        self.assertEqual('4D', cube_config.time_period)

    def test_to_dict(self):
        expected_dict = dict(variable_names=['B03', 'B04'],
                             crs='WGS84',
                             bbox=[12.2, 52.1, 13.9, 54.8],
                             spatial_res=0.05,
                             time_range=['2018-01-01', None],
                             time_period='4D')
        cube_config = CubeConfig.get_schema().from_instance(expected_dict)
        actual_dict = cube_config.to_dict()
        self.assertEqual(expected_dict, actual_dict)

        # smoke test JSON serialisation
        json.dumps(actual_dict, indent=2)


class CallbackConfigTest(unittest.TestCase):

    def test_to_dict(self):
        with self.assertRaises(ValueError) as e:
            CallbackConfig()
        self.assertEqual('Both, api_uri and access_token must be given', str(e.exception))

        expected_dict = {
            "api_uri": 'https://bla.com',
            "access_token": 'dfasovjdasoövjidfs'
        }
        callback = CallbackConfig(**expected_dict)
        actual_dict = callback.to_dict()

        self.assertDictEqual(expected_dict, actual_dict)

        # smoke test JSON serialisation
        json.dumps(actual_dict, indent=2)
