import shutil
import unittest
from io import StringIO
import os
from typing import Dict, Tuple

import yaml
# from yaml.parser import ParserError # needs to be kept for last test, which is still not working properly

from xcube.api.gen.config import get_config_dict
from xcube.util.config import flatten_dict, to_name_dict_pair, to_name_dict_pairs, to_resolved_name_dict_pairs, \
    merge_config


class ToResolvedNameDictPairsTest(unittest.TestCase):
    def test_to_resolved_name_dict_pairs_and_reject(self):
        container = ['a', 'a1', 'a2', 'b1', 'b2', 'x_c', 'y_c', 'd', 'e', 'e1', 'e_1', 'e_12', 'e_AB']
        resolved = to_resolved_name_dict_pairs([('a*', None),
                                                ('b', {'name': 'B'}),
                                                ('*_c', {'marker': True, 'name': 'C'}),
                                                ('d', {'name': 'D'}),
                                                ('e_??', {'marker': True})],
                                               container)
        self.assertEqual([('a', None),
                          ('a1', None),
                          ('a2', None),
                          # ('b', {'name': 'B'}),  # 'b' is rejected!
                          ('x_c', {'marker': True, 'name': 'C'}),
                          ('y_c', {'marker': True, 'name': 'C'}),
                          ('d', {'name': 'D'}),
                          ('e_12', {'marker': True}),
                          ('e_AB', {'marker': True})],
                         resolved)

    def test_to_resolved_name_dict_pairs_and_keep(self):
        container = ['a', 'a1', 'a2', 'b1', 'b2', 'x_c', 'y_c', 'd', 'e', 'e1', 'e_1', 'e_12', 'e_AB']
        resolved = to_resolved_name_dict_pairs([('a*', None),
                                                ('b', {'name': 'B'}),
                                                ('*_c', {'marker': True, 'name': 'C'}),
                                                ('d', {'name': 'D'}),
                                                ('e_??', {'marker': True})],
                                               container,
                                               keep=True)
        self.assertEqual([('a', None),
                          ('a1', None),
                          ('a2', None),
                          ('b', {'name': 'B'}),  # 'b' is kept!
                          ('x_c', {'marker': True, 'name': 'C'}),
                          ('y_c', {'marker': True, 'name': 'C'}),
                          ('d', {'name': 'D'}),
                          ('e_12', {'marker': True}),
                          ('e_AB', {'marker': True})],
                         resolved)


class ToNameDictPairsTest(unittest.TestCase):
    def test_to_name_dict_pairs_from_list(self):
        parent = ['a*',
                  ('b', 'B'),
                  ('*_c', {'marker': True, 'name': 'C'}),
                  {'d': 'D'},
                  {'e_??': {'marker': True}}]
        pairs = to_name_dict_pairs(parent, default_key='name')
        self.assertEqual([('a*', None),
                          ('b', {'name': 'B'}),
                          ('*_c', {'marker': True, 'name': 'C'}),
                          ('d', {'name': 'D'}),
                          ('e_??', {'marker': True})],
                         pairs)

    def test_to_name_dict_pairs_from_dict(self):
        parent = {'a*': None,
                  'b': 'B',
                  '*_c': {'marker': True, 'name': 'C'},
                  'd': 'D',
                  'e_??': {'marker': True}}
        pairs = to_name_dict_pairs(parent, default_key='name')
        self.assertEqual([('a*', None),
                          ('b', {'name': 'B'}),
                          ('*_c', {'marker': True, 'name': 'C'}),
                          ('d', {'name': 'D'}),
                          ('e_??', {'marker': True})],
                         pairs)


class ToNameDictPairTest(unittest.TestCase):
    def test_name_only(self):
        pair = to_name_dict_pair('a', default_key='name')
        self.assertEqual(('a', None), pair)
        pair = to_name_dict_pair('a?', default_key='name')
        self.assertEqual(('a?', None), pair)

        with self.assertRaises(ValueError) as cm:
            to_name_dict_pair(12, default_key='name')
        self.assertEqual('name must be a string', f'{cm.exception}')

    def test_tuple(self):
        pair = to_name_dict_pair(('a', 'udu'), default_key='name')
        self.assertEqual(('a', dict(name='udu')), pair)
        pair = to_name_dict_pair(('a*', 'udu'), default_key='name')
        self.assertEqual(('a*', dict(name='udu')), pair)
        pair = to_name_dict_pair(('a*', {'name': 'udu'}))
        self.assertEqual(('a*', dict(name='udu')), pair)
        pair = to_name_dict_pair(('a*', {'marker': True, 'name': 'udu'}))
        self.assertEqual(('a*', dict(marker=True, name='udu')), pair)

        with self.assertRaises(ValueError) as cm:
            to_name_dict_pair(('a', 5))
        self.assertEqual("value of 'a' must be a dictionary", f'{cm.exception}')

    def test_dict(self):
        pair = to_name_dict_pair('a', parent=dict(a='udu'), default_key='name')
        self.assertEqual(('a', dict(name='udu')), pair)
        pair = to_name_dict_pair('a*', parent={'a*': 'udu'}, default_key='name')
        self.assertEqual(('a*', dict(name='udu')), pair)

    def test_mapping(self):
        pair = to_name_dict_pair({'a': 'udu'}, default_key='name')
        self.assertEqual(('a', dict(name='udu')), pair)
        pair = to_name_dict_pair({'a*': dict(name='udu')})
        self.assertEqual(('a*', dict(name='udu')), pair)


class FlattenDictTest(unittest.TestCase):

    def test_flatten_dict(self):
        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            flatten_dict(None)

        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            flatten_dict(673)

        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            flatten_dict("?")

        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            flatten_dict({0: ""})

        self.assertEqual({}, flatten_dict({}))

        self.assertEqual({"title": "DCS4COP Sentinel-3 OLCI L2C Data Cube",
                          "source": "Sentinel-3 OLCI L2 surface observation"
                          },
                         flatten_dict({"title": "DCS4COP Sentinel-3 OLCI L2C Data Cube",
                                       "source": "Sentinel-3 OLCI L2 surface observation"
                                       }))

        self.assertEqual({"title": "DCS4COP Sentinel-3 OLCI L2C Data Cube",
                          "date_created": "2018-05-30",
                          "date_modified": "2018-05-30",
                          "date_issued": "2018-06-01"
                          },
                         flatten_dict({"title": "DCS4COP Sentinel-3 OLCI L2C Data Cube",
                                       "date": {
                                           "created": "2018-05-30",
                                           "modified": "2018-05-30",
                                           "issued": "2018-06-01"
                                       }}))

        self.assertEqual({"title": "DCS4COP Sentinel-3 OLCI L2C Data Cube",
                          "creator_name": "BC",
                          "creator_url": "http://www.bc.de"
                          },
                         flatten_dict({"title": "DCS4COP Sentinel-3 OLCI L2C Data Cube",
                                       "creator": [
                                           {"name": "BC",
                                            "url": "http://www.bc.de"
                                            }
                                       ]}))

        self.assertEqual({"title": "DCS4COP Sentinel-3 OLCI L2C Data Cube",
                          "creator_name": "BC, ACME",
                          "creator_url": "http://www.bc.de, http://acme.com"
                          },
                         flatten_dict({"title": "DCS4COP Sentinel-3 OLCI L2C Data Cube",
                                       "creator": [
                                           {"name": "BC",
                                            "url": "http://www.bc.de"
                                            },
                                           {"name": "ACME",
                                            "url": "http://acme.com"
                                            }
                                       ]}))

    def test_from_yaml(self):
        stream = StringIO(TEST_JSON)
        d = yaml.load(stream)
        d = flatten_dict(d['output_metadata'])
        self.assertEqual(17, len(d))
        self.assertEqual('DCS4COP Sentinel-3 OLCI L2C Data Cube',
                         d.get('title'))
        self.assertEqual('Brockmann Consult GmbH, Royal Belgian Institute for Natural Sciences (RBINS)',
                         d.get('creator_name'))
        self.assertEqual('https://www.brockmann-consult.de, http://odnature.naturalsciences.be/remsem/',
                         d.get('creator_url'))
        self.assertEqual("2018-05-30", d.get('date_created'))
        self.assertEqual("2018-06-01", d.get('date_issued'))
        self.assertEqual(0.0, d.get('geospatial_lon_min'))
        self.assertEqual(5.0, d.get('geospatial_lon_max'))
        self.assertEqual(50.0, d.get('geospatial_lat_min'))
        self.assertEqual(52.5, d.get('geospatial_lat_max'))
        self.assertEqual('degrees_east', d.get('geospatial_lon_units'))
        self.assertEqual('degrees_north', d.get('geospatial_lat_units'))
        self.assertEqual(0.0025, d.get('geospatial_lon_resolution'))
        self.assertEqual(0.0025, d.get('geospatial_lat_resolution'))
        self.assertEqual('2016-10-01', d.get('time_coverage_start'))
        self.assertEqual('2017-10-01T12:00:10', d.get('time_coverage_end'))
        self.assertEqual('P1Y', d.get('time_coverage_duration'))
        self.assertEqual('1D', d.get('time_coverage_resolution'))


TEST_JSON = """
output_metadata:
  # CF: A succinct description of what is in the dataset.
  title: "DCS4COP Sentinel-3 OLCI L2C Data Cube"
  
  # The data creator's name, URL, and email.
  # The "institution" attribute will be used if the "creator_name" attribute does not exist.
  creator:
    - name: "Brockmann Consult GmbH"
      url: "https://www.brockmann-consult.de"
    - name: "Royal Belgian Institute for Natural Sciences (RBINS)"
      url: "http://odnature.naturalsciences.be/remsem/"
  
  date:
    # The date on which the data was created.
    created:  "2018-05-30"
    # The date on which this data was formally issued.
    issued:   "2018-06-01"
  
  geospatial_lon:
    min:  0.0
    max:  5.0
    units: "degrees_east"
    resolution: 0.0025
  
  geospatial_lat:
    min: 50.0
    max: 52.5
    units: "degrees_north"
    resolution: 0.0025
  
  time_coverage:
    start:      2016-10-01
    end:        2017-10-01T12:00:10
    duration:   "P1Y"
    resolution: "1D"
"""


class MergeDictsTest(unittest.TestCase):
    def test_single_dict(self):
        first_dict = {}
        actual_dict = merge_config(first_dict)
        self.assertIs(first_dict, actual_dict)

    def test_two_dicts(self):
        first_dict = {'a': 'name'}
        second_dict = {'b': 'age'}
        actual_dict = merge_config(first_dict, second_dict)
        self.assertEqual({'a': 'name', 'b': 'age'}, actual_dict)

    def test_three_dicts(self):
        first_dict = {'a': 'name'}
        second_dict = {'b': 'age'}
        third_dict = {'c': 'hair'}
        actual_dict = merge_config(first_dict, second_dict, third_dict)
        self.assertEqual({'a': 'name', 'b': 'age', 'c': 'hair'}, actual_dict)

    def test_order_matters(self):
        first_dict = {'a': 'name', 'b': 42}
        sec_dict = {'a': 'age'}
        third_dict = {'b': 15}
        actual_dict = merge_config(first_dict, sec_dict, third_dict)
        self.assertEqual({'a': 'age', 'b': 15}, actual_dict)

    def test_merge_deep(self):
        first_dict = {'a': dict(b=42, c=15), 'o': 105}
        second_dict = {'a': dict(c=25)}
        actual_dict = merge_config(first_dict, second_dict)
        self.assertEqual({'a': dict(b=42, c=25), 'o': 105}, actual_dict)

    def test_merge_dict_value(self):
        first_dict = {'o': 105}
        second_dict = {'a': dict(c=25)}
        actual_dict = merge_config(first_dict, second_dict)
        self.assertEqual({'a': dict(c=25), 'o': 105}, actual_dict)


def _get_config_obj(config_file: Tuple[str, str] = (),
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


TEMP_PATH_FOR_YAML = './temp_test_data_for_xcube_tests'

CONFIG_1_NAME = 'config_1.json'
CONFIG_2_NAME = 'config_2.json'
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

CONFIG_3_NAME = 'config_3.json'
CONFIG_4_NAME = 'config_4.json'
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
                yaml.dump(yaml.load(config_yaml), outfile)
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
            config_obj = _get_config_obj(config_file=CONFIG_1_FILE_LIST)
            _create_temp_yaml(TEMP_PATH_FOR_YAML, CONFIG_1_NAME, config_obj)

            config = get_config_dict(config_obj)
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
            config_obj = _get_config_obj(config_file=CONFIG_1_FILE_LIST,
                                         output_variables='a,b')
            _create_temp_yaml(TEMP_PATH_FOR_YAML, CONFIG_1_NAME, config_obj)
            config = get_config_dict(config_obj)
            self.assertIn('output_variables', config)
            self.assertIsNotNone(['a', 'b'], config['output_variables'])
        finally:
            if os.path.exists(TEMP_PATH_FOR_YAML):
                shutil.rmtree(TEMP_PATH_FOR_YAML)
                print('Successfully removed folder')

    def test_config_file_does_not_exist(self):
        config_obj = _get_config_obj(config_file=['bibo.yaml', ])
        with self.assertRaises(FileNotFoundError) as cm:
            get_config_dict(config_obj)
        self.assertEqual("[Errno 2] No such file or directory: 'bibo.yaml'",
                         f'{cm.exception}')

    def test_output_size_option(self):
        config_obj = _get_config_obj(output_size='120, 140')
        config = get_config_dict(config_obj)
        self.assertIn('output_size', config)
        self.assertEqual([120, 140], config['output_size'])

        config_obj = _get_config_obj(output_size='120,abc')
        with self.assertRaises(ValueError) as cm:
            get_config_dict(config_obj)
        self.assertEqual(
            "Invalid output size was given. Only integers are accepted. The given output size was: '120,abc'",
            f'{cm.exception}')

    def test_output_region_option(self):
        config_obj = _get_config_obj(output_region='-10.5, 5., 10.5, 25.')
        config = get_config_dict(config_obj)
        self.assertIn('output_region', config)
        self.assertEqual([-10.5, 5., 10.5, 25.], config['output_region'])

        config_obj = _get_config_obj(output_region='50,_2,55,21')
        with self.assertRaises(ValueError) as cm:
            get_config_dict(config_obj)
        self.assertEqual("Invalid output region was given. Only floats are accepted. The given output region was:"
                         " '50,_2,55,21'",
                         f'{cm.exception}')

        config_obj = _get_config_obj(output_region='50, 20, 55')
        with self.assertRaises(ValueError) as cm:
            get_config_dict(config_obj)
        self.assertEqual("The output region must be given as 4 values: <lon_min>,<lat_min>,<lon_max>,<lat_max>, "
                         "but was: '50, 20, 55'",
                         f'{cm.exception}')

    def test_output_variables_option(self):
        config_obj = _get_config_obj(output_variables='hanni, nanni, pfanni')
        config = get_config_dict(config_obj)
        self.assertIn('output_variables', config)
        self.assertEqual([('hanni', None), ('nanni', None), ('pfanni', None)],
                         config['output_variables'])

        config_obj = _get_config_obj(output_variables='')
        with self.assertRaises(ValueError) as cm:
            get_config_dict(config_obj)
        self.assertEqual("output_variables must contain at least one name",
                         f'{cm.exception}')

        config_obj = _get_config_obj(output_variables='a*,')
        with self.assertRaises(ValueError) as cm:
            get_config_dict(config_obj)
        self.assertEqual("all names in output_variables must be non-empty",
                         f'{cm.exception}')

    # This test is still not running correcly, needs to be fixed. TODO: AliceBalfanz
    # def test_config_file_with_invalid_yaml(self):
    #     try:
    #         _create_temp_yaml(TEMP_PATH_FOR_YAML, CONFIG_4_NAME, CONFIG_2_YAML)
    #         config_obj = _get_config_obj(config_file=CONFIG_2_FILE_LIST)
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
