import unittest

from xcube.cli._gen2.request import CubeConfig
from xcube.cli._gen2.request import InputConfig
from xcube.cli._gen2.request import OutputConfig
from xcube.cli._gen2.request import Request


class InputConfigTest(unittest.TestCase):

    def test_from_json_instance(self):
        json_instance = dict(cube_store_id='mem', cube_id='S2L2A', variable_names=['B01', 'B02'])
        input_config = InputConfig.get_schema().from_instance(json_instance)
        self.assertIsInstance(input_config, InputConfig)
        self.assertEqual('mem', input_config.cube_store_id)
        self.assertEqual('S2L2A', input_config.cube_id)
        self.assertEqual(['B01', 'B02'], input_config.variable_names)
        self.assertEqual(None, input_config.cube_store_params)
        self.assertEqual(None, input_config.open_params)


class OutputConfigTest(unittest.TestCase):

    def test_from_json_instance(self):
        json_instance = dict(cube_store_id='mem', cube_id='CHL')
        output_config = OutputConfig.get_schema().from_instance(json_instance)
        self.assertIsInstance(output_config, OutputConfig)
        self.assertEqual('mem', output_config.cube_store_id)
        self.assertEqual('CHL', output_config.cube_id)
        self.assertEqual(None, output_config.cube_store_params)
        self.assertEqual(None, output_config.write_params)


class CubeConfigTest(unittest.TestCase):

    def test_from_json_instance(self):
        json_instance = dict(spatial_crs='WGS84',
                             spatial_coverage=[12.2, 52.1, 13.9, 54.8],
                             spatial_resolution=0.05,
                             temporal_coverage=['2018-01-01', None],
                             temporal_resolution='4D')
        cube_config = CubeConfig.get_schema().from_instance(json_instance)
        self.assertIsInstance(cube_config, CubeConfig)
        self.assertEqual('WGS84', cube_config.spatial_crs)
        self.assertEqual((12.2, 52.1, 13.9, 54.8), cube_config.spatial_coverage)
        self.assertEqual(0.05, cube_config.spatial_resolution)
        self.assertEqual(('2018-01-01', None), cube_config.temporal_coverage)
        self.assertEqual('4D', cube_config.temporal_resolution)


class RequestTest(unittest.TestCase):

    def test_from_dict(self):
        request_dict = dict(input_configs=[dict(cube_store_id='mem',
                                                cube_id='S2L2A',
                                                variable_names=['B01', 'B02'])],
                            cube_config=dict(spatial_crs='WGS84',
                                             spatial_coverage=[12.2, 52.1, 13.9, 54.8],
                                             spatial_resolution=0.05,
                                             temporal_coverage=['2018-01-01', None],
                                             temporal_resolution='4D'),
                            output_config=dict(cube_store_id='mem',
                                               cube_id='CHL'))
        request = Request.from_dict(request_dict)
        self.assertIsInstance(request, Request)
        self.assertEqual(1, len(request.input_configs))
        self.assertIsInstance(request.input_configs[0], InputConfig)
        self.assertEqual('mem', request.input_configs[0].cube_store_id)
        self.assertEqual('S2L2A', request.input_configs[0].cube_id)
        self.assertEqual(['B01', 'B02'], request.input_configs[0].variable_names)
        self.assertIsInstance(request.output_config, OutputConfig)
        self.assertEqual('mem', request.output_config.cube_store_id)
        self.assertEqual('CHL', request.output_config.cube_id)
        self.assertIsInstance(request.cube_config, CubeConfig)
        self.assertEqual('WGS84', request.cube_config.spatial_crs)
        self.assertEqual((12.2, 52.1, 13.9, 54.8), request.cube_config.spatial_coverage)
        self.assertEqual(0.05, request.cube_config.spatial_resolution)
        self.assertEqual(('2018-01-01', None), request.cube_config.temporal_coverage)
        self.assertEqual('4D', request.cube_config.temporal_resolution)
