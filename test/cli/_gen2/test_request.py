import unittest

from xcube.cli._gen2.genconfig import CubeConfig
from xcube.cli._gen2.genconfig import InputConfig
from xcube.cli._gen2.genconfig import OutputConfig
from xcube.cli._gen2.genconfig import GenConfig


class InputConfigTest(unittest.TestCase):

    def test_from_json_instance(self):
        json_instance = dict(store_id='sentinelhub', data_id='S2L2A', variable_names=['B01', 'B02'])
        input_config = InputConfig.get_schema().from_instance(json_instance)
        self.assertIsInstance(input_config, InputConfig)
        self.assertEqual('sentinelhub', input_config.store_id)
        self.assertEqual('S2L2A', input_config.data_id)
        self.assertEqual(['B01', 'B02'], input_config.variable_names)
        self.assertEqual({}, input_config.store_params)
        self.assertEqual({}, input_config.open_params)


class OutputConfigTest(unittest.TestCase):

    def test_from_json_instance(self):
        json_instance = dict(store_id='s3', data_id='CHL.zarr')
        output_config = OutputConfig.get_schema().from_instance(json_instance)
        self.assertIsInstance(output_config, OutputConfig)
        self.assertEqual('s3', output_config.store_id)
        self.assertEqual('CHL.zarr', output_config.data_id)
        self.assertEqual({}, output_config.store_params)
        self.assertEqual({}, output_config.write_params)


class CubeConfigTest(unittest.TestCase):

    def test_from_json_instance(self):
        json_instance = dict(crs='WGS84',
                             bbox=[12.2, 52.1, 13.9, 54.8],
                             spatial_res=0.05,
                             time_range=['2018-01-01', None],
                             time_period='4D')
        cube_config = CubeConfig.get_schema().from_instance(json_instance)
        self.assertIsInstance(cube_config, CubeConfig)
        self.assertEqual('WGS84', cube_config.crs)
        self.assertEqual((12.2, 52.1, 13.9, 54.8), cube_config.bbox)
        self.assertEqual(0.05, cube_config.spatial_res)
        self.assertEqual(('2018-01-01', None), cube_config.time_range)
        self.assertEqual('4D', cube_config.time_period)

    def test_to_dict(self):
        json_instance = dict(crs='WGS84',
                             bbox=[12.2, 52.1, 13.9, 54.8],
                             spatial_res=0.05,
                             time_range=['2018-01-01', None],
                             time_period='4D')
        cube_config = CubeConfig.get_schema().from_instance(json_instance)
        self.assertEqual(json_instance, cube_config.to_dict())


class RequestTest(unittest.TestCase):

    def test_from_dict(self):
        request_dict = dict(input_configs=[dict(store_id='memory',
                                                data_id='S2L2A',
                                                variable_names=['B01', 'B02'])],
                            cube_config=dict(crs='WGS84',
                                             bbox=[12.2, 52.1, 13.9, 54.8],
                                             spatial_res=0.05,
                                             time_range=['2018-01-01', None],
                                             time_period='4D'),
                            output_config=dict(store_id='memory',
                                               data_id='CHL'))
        request = GenConfig.from_dict(request_dict)
        self.assertIsInstance(request, GenConfig)
        self.assertEqual(1, len(request.input_configs))
        self.assertIsInstance(request.input_configs[0], InputConfig)
        self.assertEqual('memory', request.input_configs[0].store_id)
        self.assertEqual('S2L2A', request.input_configs[0].data_id)
        self.assertEqual(['B01', 'B02'], request.input_configs[0].variable_names)
        self.assertIsInstance(request.output_config, OutputConfig)
        self.assertEqual('memory', request.output_config.store_id)
        self.assertEqual('CHL', request.output_config.data_id)
        self.assertIsInstance(request.cube_config, CubeConfig)
        self.assertEqual('WGS84', request.cube_config.crs)
        self.assertEqual((12.2, 52.1, 13.9, 54.8), request.cube_config.bbox)
        self.assertEqual(0.05, request.cube_config.spatial_res)
        self.assertEqual(('2018-01-01', None), request.cube_config.time_range)
        self.assertEqual('4D', request.cube_config.time_period)

    def test_to_dict(self):
        request_dict = dict(input_configs=[dict(store_id='memory',
                                                data_id='S2L2A',
                                                variable_names=['B01', 'B02'])],
                            cube_config=dict(crs='WGS84',
                                             bbox=[12.2, 52.1, 13.9, 54.8],
                                             spatial_res=0.05,
                                             time_range=['2018-01-01', None],
                                             time_period='4D'),
                            output_config=dict(store_id='memory',
                                               data_id='CHL'))
        request = GenConfig.from_dict(request_dict)
        self.assertEqual(request_dict, request.to_dict())
        # import json
        # print(json.dumps(request.to_dict()))
