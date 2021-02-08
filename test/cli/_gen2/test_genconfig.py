import unittest

from xcube.cli._gen2.genconfig import CubeConfig
from xcube.cli._gen2.genconfig import GenConfig
from xcube.cli._gen2.genconfig import InputConfig
from xcube.cli._gen2.genconfig import OutputConfig


class InputConfigTest(unittest.TestCase):

    def test_from_json_instance(self):
        json_instance = dict(store_id='sentinelhub', data_id='S2L2A')
        input_config = InputConfig.get_schema().from_instance(json_instance)
        self.assertIsInstance(input_config, InputConfig)
        self.assertEqual('sentinelhub', input_config.store_id)
        self.assertEqual('S2L2A', input_config.data_id)
        self.assertEqual(None, input_config.store_params)
        self.assertEqual(None, input_config.open_params)


class OutputConfigTest(unittest.TestCase):

    def test_from_json_instance(self):
        json_instance = dict(store_id='s3', data_id='CHL.zarr')
        output_config = OutputConfig.get_schema().from_instance(json_instance)
        self.assertIsInstance(output_config, OutputConfig)
        self.assertEqual('s3', output_config.store_id)
        self.assertEqual('CHL.zarr', output_config.data_id)
        self.assertEqual(None, output_config.store_params)
        self.assertEqual(None, output_config.write_params)


class CubeConfigTest(unittest.TestCase):

    def test_from_json_instance(self):
        json_instance = dict(variable_names=['B03', 'B04'],
                             crs='WGS84',
                             bbox=[12.2, 52.1, 13.9, 54.8],
                             spatial_res=0.05,
                             time_range=['2018-01-01', None],
                             time_period='4D')
        cube_config = CubeConfig.get_schema().from_instance(json_instance)
        self.assertIsInstance(cube_config, CubeConfig)
        self.assertEqual(('B03', 'B04'), cube_config.variable_names)
        self.assertEqual('WGS84', cube_config.crs)
        self.assertEqual((12.2, 52.1, 13.9, 54.8), cube_config.bbox)
        self.assertEqual(0.05, cube_config.spatial_res)
        self.assertEqual(('2018-01-01', None), cube_config.time_range)
        self.assertEqual('4D', cube_config.time_period)

    def test_to_dict(self):
        json_instance = dict(variable_names=['B03', 'B04'],
                             crs='WGS84',
                             bbox=[12.2, 52.1, 13.9, 54.8],
                             spatial_res=0.05,
                             time_range=['2018-01-01', None],
                             time_period='4D')
        cube_config = CubeConfig.get_schema().from_instance(json_instance)
        self.assertEqual(json_instance, cube_config.to_dict())


class GenConfigTest(unittest.TestCase):

    def test_from_dict(self):
        request_dict = dict(input_configs=[dict(store_id='memory',
                                                data_id='S2L2A')],
                            cube_config=dict(variable_names=['B01', 'B02'],
                                             crs='WGS84',
                                             bbox=[12.2, 52.1, 13.9, 54.8],
                                             spatial_res=0.05,
                                             time_range=['2018-01-01', None],
                                             time_period='4D'),
                            output_config=dict(store_id='memory',
                                               data_id='CHL'))
        gen_config = GenConfig.from_dict(request_dict)
        self.assertIsInstance(gen_config, GenConfig)
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
        request_dict = dict(input_config=dict(store_id='memory',
                                              data_id='S2L2A'),
                            cube_config=dict(variable_names=['B01', 'B02'],
                                             crs='WGS84',
                                             bbox=[12.2, 52.1, 13.9, 54.8],
                                             spatial_res=0.05,
                                             time_range=['2018-01-01', None],
                                             time_period='4D'),
                            output_config=dict(store_id='memory',
                                               data_id='CHL'))
        request = GenConfig.from_dict(request_dict)
        self.assertEqual(request_dict, request.to_dict())
