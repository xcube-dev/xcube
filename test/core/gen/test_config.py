import os
import shutil
import unittest

import yaml

from xcube.core.gen.config import get_config_dict

TEMP_PATH_FOR_YAML = './temp_test_data_for_xcube_tests'

CONFIG_1_NAME = 'config_1.yaml'
CONFIG_2_NAME = 'config_2.yaml'
CONFIG_1_FILE_LIST = [(os.path.join(TEMP_PATH_FOR_YAML, CONFIG_1_NAME)),
                      (os.path.join(TEMP_PATH_FOR_YAML, CONFIG_2_NAME))]
CONFIG_1_YAML = """
output_size: [2000, 1000] 
output_region: [0, 20, 20, 30] 
output_variables: 
  - x
  - y
  - z*
"""

CONFIG_3_NAME = 'config_3.yaml'
CONFIG_4_NAME = 'config_4.yaml'
CONFIG_2_FILE_LIST = [(os.path.join(TEMP_PATH_FOR_YAML, CONFIG_3_NAME)),
                      (os.path.join(TEMP_PATH_FOR_YAML, CONFIG_4_NAME))]
CONFIG_2_YAML = """
: output_variables: 
  - x
 6--
"""


def _create_temp_yaml(temp_path_for_yaml, config_file_name, config_yaml):
    if not os.path.exists(TEMP_PATH_FOR_YAML):
        try:
            os.mkdir(os.path.join(temp_path_for_yaml))
        except OSError as e:
            print(e)
            print("Creation of the directory %s failed" % temp_path_for_yaml)
        else:
            print("Successfully created the directory %s " % temp_path_for_yaml)
            yaml_path = os.path.join(temp_path_for_yaml, config_file_name)
            with open(yaml_path, 'w') as outfile:
                yaml.dump(yaml.full_load(config_yaml), outfile)
            return yaml_path

    else:
        yaml_path = os.path.join(temp_path_for_yaml, config_file_name)
        with open(yaml_path, 'w') as outfile:
            yaml.dump(config_yaml, outfile, default_flow_style=False)
        return yaml_path


class GetConfigDictTest(unittest.TestCase):
    def test_config_file_alone(self):
        try:
            _create_temp_yaml(TEMP_PATH_FOR_YAML, CONFIG_2_NAME, CONFIG_1_YAML)
            config_obj = dict(config_files=CONFIG_1_FILE_LIST)
            _create_temp_yaml(TEMP_PATH_FOR_YAML, CONFIG_1_NAME, config_obj)

            config = get_config_dict(**config_obj)
            self.assertIsNotNone(config)
            self.assertEqual([2000, 1000], config['output_size'])
            self.assertEqual([0, 20, 20, 30], config['output_region'])
            self.assertEqual([('x', None), ('y', None), ('z*', None)], config['output_variables'])
        finally:
            if os.path.exists(TEMP_PATH_FOR_YAML):
                shutil.rmtree(TEMP_PATH_FOR_YAML)
                print('Successfully removed folder')

    def test_config_file_overwritten_by_config_obj(self):
        try:
            _create_temp_yaml(TEMP_PATH_FOR_YAML, CONFIG_2_NAME, CONFIG_1_YAML)
            config_obj = dict(config_files=CONFIG_1_FILE_LIST,
                              output_variables='a,b')
            _create_temp_yaml(TEMP_PATH_FOR_YAML, CONFIG_1_NAME, config_obj)
            config = get_config_dict(**config_obj)
            self.assertIn('output_variables', config)
            self.assertIsNotNone(['a', 'b'], config['output_variables'])
        finally:
            if os.path.exists(TEMP_PATH_FOR_YAML):
                shutil.rmtree(TEMP_PATH_FOR_YAML)
                print('Successfully removed folder')

    def test_config_file_does_not_exist(self):
        config_obj = dict(config_files=['bibo.yaml', ])
        with self.assertRaises(ValueError) as cm:
            get_config_dict(**config_obj)
        self.assertEqual("Cannot find configuration 'bibo.yaml'",
                         f'{cm.exception}')

    def test_output_size_option(self):
        config_obj = dict(output_size='120, 140')
        config = get_config_dict(**config_obj)
        self.assertIn('output_size', config)
        self.assertEqual([120, 140], config['output_size'])

        config_obj = dict(output_size='120,abc')
        with self.assertRaises(ValueError) as cm:
            get_config_dict(**config_obj)
        self.assertEqual(
            "output_size must have the form <width>,<height>, where both values must be positive integer numbers",
            f'{cm.exception}')

    def test_output_region_option(self):
        config_obj = dict(output_region='-10.5, 5., 10.5, 25.')
        config = get_config_dict(**config_obj)
        self.assertIn('output_region', config)
        self.assertEqual([-10.5, 5., 10.5, 25.], config['output_region'])

        config_obj = dict(output_region='50,_2,55,21')
        with self.assertRaises(ValueError) as cm:
            get_config_dict(**config_obj)
        self.assertEqual("output_region must have the form <lon_min>,<lat_min>,<lon_max>,<lat_max>,"
                         " where all four numbers must be floating point numbers in degrees",
                         f'{cm.exception}')

        config_obj = dict(output_region='50, 20, 55')
        with self.assertRaises(ValueError) as cm:
            get_config_dict(**config_obj)
        self.assertEqual("output_region must have the form <lon_min>,<lat_min>,<lon_max>,<lat_max>,"
                         " where all four numbers must be floating point numbers in degrees",
                         f'{cm.exception}')

    def test_output_variables_option(self):
        config_obj = dict(output_variables='hanni, nanni, pfanni')
        config = get_config_dict(**config_obj)
        self.assertIn('output_variables', config)
        self.assertEqual([('hanni', None), ('nanni', None), ('pfanni', None)],
                         config['output_variables'])

        config_obj = dict(output_variables='')
        with self.assertRaises(ValueError) as cm:
            get_config_dict(**config_obj)
        self.assertEqual("output_variables must be a list of existing variable names",
                         f'{cm.exception}')

        config_obj = dict(output_variables='a*,')
        with self.assertRaises(ValueError) as cm:
            get_config_dict(**config_obj)
        self.assertEqual("output_variables must be a list of existing variable names",
                         f'{cm.exception}')

    # This test is still not running correcly, needs to be fixed. TODO: AliceBalfanz
    # def test_config_file_with_invalid_yaml(self):
    #     try:
    #         _create_temp_yaml(TEMP_PATH_FOR_YAML, CONFIG_4_NAME, CONFIG_2_YAML)
    #         config_obj = dict(config_files=CONFIG_2_FILE_LIST)
    #         _create_temp_yaml(TEMP_PATH_FOR_YAML, CONFIG_3_NAME, config_obj)
    #
    #         with self.assertRaises(ParserError) as cm:
    #             get_config_dict(config_obj)
    #         self.assertEqual('YAML in \'config_2.json\' is invalid: '
    #                          'while parsing a block mapping\n'
    #                          'expected <block end>, but found \':\'\n'
    #                          '  in "<file>", line 2, column 1',
    #                          f'{cm.exception}')
    #     finally:
    #         if os.path.exists(TEMP_PATH_FOR_YAML):
    #             shutil.rmtree(TEMP_PATH_FOR_YAML)
    #             print('Successfully removed folder')
