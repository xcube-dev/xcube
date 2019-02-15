import os
import shutil
import unittest
from typing import Dict

import yaml

from xcube.api.gen.config import get_config_dict


def get_config_obj(config_file=None,
                   input_files=None,
                   input_processor=None,
                   output_dir=None,
                   output_name=None,
                   output_writer=None,
                   output_size=None,
                   output_region=None,
                   output_variables=None,
                   output_resampling=None) -> Dict:
    return dict(config_file=config_file,
                input_files=input_files,
                input_processor=input_processor,
                output_dir=output_dir,
                output_name=output_name,
                output_writer=output_writer,
                output_size=output_size,
                output_region=output_region,
                output_variables=output_variables,
                output_resampling=output_resampling)


TEMP_PATH_FOR_YAML = 'temp_test_data_for_xcube_tests'

CONFIG_1_FILE = 'config_1.json'
CONFIG_1_FILE_2 = 'config_1_2.json'
CONFIG_1_FILE_LIST = [(os.path.join(os.path.expanduser('~'), TEMP_PATH_FOR_YAML, CONFIG_1_FILE)),
                      (os.path.join(os.path.expanduser('~'), TEMP_PATH_FOR_YAML, CONFIG_1_FILE_2))]
CONFIG_1_YAML = """
output_size: [2000, 1000] 
output_region: [0, 20, 20, 30] 
output_variables: 
  - x
  - y
  - z*
"""


def _create_temp_yaml(temp_path_for_yaml, config_file_name, config_yaml):
    if not os.path.exists(os.path.join(os.path.expanduser('~'), TEMP_PATH_FOR_YAML)):
        try:
            os.mkdir(os.path.join(os.path.expanduser('~'), temp_path_for_yaml))
        except OSError as e:
            print(e)
            print("Creation of the directory %s failed" % temp_path_for_yaml)
        else:
            print("Successfully created the directory %s " % temp_path_for_yaml)
            yaml_path = os.path.join(os.path.expanduser('~'), temp_path_for_yaml, config_file_name)
            with open(yaml_path, 'w') as outfile:
                yaml.dump(yaml.load(config_yaml), outfile)
            return yaml_path

    else:
        yaml_path = os.path.join(os.path.expanduser('~'), temp_path_for_yaml, config_file_name)
        with open(yaml_path, 'w') as outfile:
            yaml.dump(config_yaml, outfile, default_flow_style=False)
        return yaml_path


class GetConfigDictTest(unittest.TestCase):
    def test_config_file_alone(self):
        try:
            _create_temp_yaml(TEMP_PATH_FOR_YAML, CONFIG_1_FILE_2, CONFIG_1_YAML)
            config_obj = get_config_obj(config_file=CONFIG_1_FILE_LIST)
            _create_temp_yaml(TEMP_PATH_FOR_YAML, CONFIG_1_FILE, config_obj)

            config = get_config_dict(config_obj)
            self.assertIsNotNone(config)
            self.assertEqual([2000, 1000], config['output_size'])
            self.assertEqual([0, 20, 20, 30], config['output_region'])
            self.assertEqual([('x', None), ('y', None), ('z*', None)], config['output_variables'])
        finally:
            if os.path.exists(os.path.join(os.path.expanduser('~'), TEMP_PATH_FOR_YAML)):
                shutil.rmtree(os.path.join(os.path.expanduser('~'), TEMP_PATH_FOR_YAML))
                print('Successfully removed folder')

    def test_config_file_overwritten_by_config_obj(self):
        try:
            _create_temp_yaml(TEMP_PATH_FOR_YAML, CONFIG_1_FILE_2, CONFIG_1_YAML)
            config_obj = get_config_obj(config_file=CONFIG_1_FILE_LIST,
                                        output_variables='a,b')
            _create_temp_yaml(TEMP_PATH_FOR_YAML, CONFIG_1_FILE, config_obj)
            config = get_config_dict(config_obj)
            self.assertIn('output_variables', config)
            self.assertIsNotNone(['a', 'b'], config['output_variables'])
        finally:
            if os.path.exists(os.path.join(os.path.expanduser('~'), TEMP_PATH_FOR_YAML)):
                shutil.rmtree(os.path.join(os.path.expanduser('~'), TEMP_PATH_FOR_YAML))
                print('Successfully removed folder')

    def test_config_file_does_not_exist(self):
        config_obj = get_config_obj(config_file=['bibo.yaml', ])
        with self.assertRaises(FileNotFoundError) as cm:
            get_config_dict(config_obj)
        self.assertEqual("[Errno 2] No such file or directory: 'bibo.yaml'",
                         f'{cm.exception}')

    def test_output_size_option(self):
        config_obj = get_config_obj(output_size='120, 140')
        config = get_config_dict(config_obj)
        self.assertIn('output_size', config)
        self.assertEqual([120, 140], config['output_size'])

        config_obj = get_config_obj(output_size='120,abc')
        with self.assertRaises(ValueError) as cm:
            get_config_dict(config_obj)
        self.assertEqual("invalid output_size '120,abc'",
                         f'{cm.exception}')

    def test_output_region_option(self):
        config_obj = get_config_obj(output_region='-10.5, 5., 10.5, 25.')
        config = get_config_dict(config_obj)
        self.assertIn('output_region', config)
        self.assertEqual([-10.5, 5., 10.5, 25.], config['output_region'])

        config_obj = get_config_obj(output_region='50,_2,55,21')
        with self.assertRaises(ValueError) as cm:
            get_config_dict(config_obj)
        self.assertEqual("invalid output_region '50,_2,55,21'",
                         f'{cm.exception}')

        config_obj = get_config_obj(output_region='50, 20, 55')
        with self.assertRaises(ValueError) as cm:
            get_config_dict(config_obj)
        self.assertEqual("invalid output_region '50, 20, 55'",
                         f'{cm.exception}')

    def test_output_variables_option(self):
        config_obj = get_config_obj(output_variables='hanni, nanni, pfanni')
        config = get_config_dict(config_obj)
        self.assertIn('output_variables', config)
        self.assertEqual([('hanni', None), ('nanni', None), ('pfanni', None)],
                         config['output_variables'])

        config_obj = get_config_obj(output_variables='')
        with self.assertRaises(ValueError) as cm:
            get_config_dict(config_obj)
        self.assertEqual("output_variables must contain at least one name",
                         f'{cm.exception}')

        config_obj = get_config_obj(output_variables='a*,')
        with self.assertRaises(ValueError) as cm:
            get_config_dict(config_obj)
        self.assertEqual("all names in output_variables must be non-empty",
                         f'{cm.exception}')
