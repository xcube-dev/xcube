import unittest
from collections import namedtuple
from io import StringIO

from xcube.api.gen.config import get_config_dict

ConfigObj = namedtuple('ConfigObj',
                       ['config_file',
                        'input_files',
                        'input_processor',
                        'output_dir',
                        'output_name',
                        'output_writer',
                        'output_size',
                        'output_region',
                        'output_variables',
                        'output_resampling'])


def get_config_obj(config_file=None,
                   input_files=None,
                   input_processor=None,
                   output_dir=None,
                   output_name=None,
                   output_writer=None,
                   output_size=None,
                   output_region=None,
                   output_variables=None,
                   output_resampling=None) -> ConfigObj:
    return ConfigObj(config_file=config_file,
                     input_files=input_files,
                     input_processor=input_processor,
                     output_dir=output_dir,
                     output_name=output_name,
                     output_writer=output_writer,
                     output_size=output_size,
                     output_region=output_region,
                     output_variables=output_variables,
                     output_resampling=output_resampling)


CONFIG_1_FILE = 'config_1.json'
CONFIG_1_YAML = """
output_size: [2000, 1000] 
output_region: [0, 20, 20, 30] 
output_variables: 
  - x
  - y
  - z*
"""

CONFIG_2_FILE = 'config_2.json'
CONFIG_2_YAML = """
: output_variables: 
  - x
 6--
"""

CONFIG_FILES = {
    CONFIG_1_FILE: CONFIG_1_YAML,
    CONFIG_2_FILE: CONFIG_2_YAML,
}


def _test_open(config_file):
    if config_file in CONFIG_FILES:
        return StringIO(CONFIG_FILES[config_file])
    else:
        raise OSError('file not found')


class GetConfigDictTest(unittest.TestCase):
    def test_config_file_alone(self):
        config_obj = get_config_obj(config_file=CONFIG_1_FILE)
        config = get_config_dict(config_obj, _test_open)
        self.assertIsNotNone(config)
        self.assertEqual([2000, 1000], config['output_size'])
        self.assertEqual([0, 20, 20, 30], config['output_region'])
        self.assertEqual([('x', None), ('y', None), ('z*', None)], config['output_variables'])

    def test_config_file_overwritten_by_config_obj(self):
        config_obj = get_config_obj(config_file=CONFIG_1_FILE,
                                    output_variables='a,b')
        config = get_config_dict(config_obj, _test_open)
        self.assertIn('output_variables', config)
        self.assertIsNotNone(['a', 'b'], config['output_variables'])

    def test_config_file_with_invalid_yaml(self):
        config_obj = get_config_obj(config_file=CONFIG_2_FILE)
        with self.assertRaises(ValueError) as cm:
            get_config_dict(config_obj, _test_open)
        self.assertEqual('YAML in \'config_2.json\' is invalid: '
                         'while parsing a block mapping\n'
                         'expected <block end>, but found \':\'\n'
                         '  in "<file>", line 2, column 1',
                         f'{cm.exception}')

    def test_config_file_does_not_exist(self):
        config_obj = get_config_obj(config_file='bibo.yaml')
        with self.assertRaises(ValueError) as cm:
            get_config_dict(config_obj, _test_open)
        self.assertEqual("cannot load configuration from 'bibo.yaml': file not found",
                         f'{cm.exception}')

    def test_output_size_option(self):
        config_obj = get_config_obj(output_size='120, 140')
        config = get_config_dict(config_obj, _test_open)
        self.assertIn('output_size', config)
        self.assertEqual([120, 140], config['output_size'])

        config_obj = get_config_obj(output_size='120,abc')
        with self.assertRaises(ValueError) as cm:
            get_config_dict(config_obj, _test_open)
        self.assertEqual("invalid output_size '120,abc'",
                         f'{cm.exception}')

    def test_output_region_option(self):
        config_obj = get_config_obj(output_region='-10.5, 5., 10.5, 25.')
        config = get_config_dict(config_obj, _test_open)
        self.assertIn('output_region', config)
        self.assertEqual([-10.5, 5., 10.5, 25.], config['output_region'])

        config_obj = get_config_obj(output_region='50,_2,55,21')
        with self.assertRaises(ValueError) as cm:
            get_config_dict(config_obj, _test_open)
        self.assertEqual("invalid output_region '50,_2,55,21'",
                         f'{cm.exception}')

        config_obj = get_config_obj(output_region='50, 20, 55')
        with self.assertRaises(ValueError) as cm:
            get_config_dict(config_obj, _test_open)
        self.assertEqual("invalid output_region '50, 20, 55'",
                         f'{cm.exception}')

    def test_output_variables_option(self):
        config_obj = get_config_obj(output_variables='hanni, nanni, pfanni')
        config = get_config_dict(config_obj, _test_open)
        self.assertIn('output_variables', config)
        self.assertEqual([('hanni', None), ('nanni', None), ('pfanni', None)],
                         config['output_variables'])

        config_obj = get_config_obj(output_variables='')
        with self.assertRaises(ValueError) as cm:
            get_config_dict(config_obj, _test_open)
        self.assertEqual("output_variables must contain at least one name",
                         f'{cm.exception}')

        config_obj = get_config_obj(output_variables='a*,')
        with self.assertRaises(ValueError) as cm:
            get_config_dict(config_obj, _test_open)
        self.assertEqual("all names in output_variables must be non-empty",
                         f'{cm.exception}')
