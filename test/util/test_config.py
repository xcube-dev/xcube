import unittest
from io import StringIO

import fsspec
import pytest
import yaml

from xcube.util.config import flatten_dict
from xcube.util.config import load_configs
from xcube.util.config import merge_config
from xcube.util.config import to_name_dict_pair
from xcube.util.config import to_name_dict_pairs
from xcube.util.config import to_resolved_name_dict_pairs


# from yaml.parser import ParserError # needs to be kept for last test, which is still not working properly


class ToResolvedNameDictPairsTest(unittest.TestCase):
    def test_to_resolved_name_dict_pairs_and_reject(self):
        container = ['a', 'a1', 'a2', 'b1', 'b2', 'x_c', 'y_c', 'd', 'e', 'e1',
                     'e_1', 'e_12', 'e_AB']
        resolved = to_resolved_name_dict_pairs([('a*', None),
                                                ('b', {'name': 'B'}),
                                                ('*_c',
                                                 {'marker': True, 'name': 'C'}),
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
        container = ['a', 'a1', 'a2', 'b1', 'b2', 'x_c', 'y_c', 'd', 'e', 'e1',
                     'e_1', 'e_12', 'e_AB']
        resolved = to_resolved_name_dict_pairs([('a*', None),
                                                ('b', {'name': 'B'}),
                                                ('*_c',
                                                 {'marker': True, 'name': 'C'}),
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
                         flatten_dict(
                             {"title": "DCS4COP Sentinel-3 OLCI L2C Data Cube",
                              "source": "Sentinel-3 OLCI L2 surface observation"
                              }))

        self.assertEqual({"title": "DCS4COP Sentinel-3 OLCI L2C Data Cube",
                          "date_created": "2018-05-30",
                          "date_modified": "2018-05-30",
                          "date_issued": "2018-06-01"
                          },
                         flatten_dict(
                             {"title": "DCS4COP Sentinel-3 OLCI L2C Data Cube",
                              "date": {
                                  "created": "2018-05-30",
                                  "modified": "2018-05-30",
                                  "issued": "2018-06-01"
                              }}))

        self.assertEqual({"title": "DCS4COP Sentinel-3 OLCI L2C Data Cube",
                          "creator_name": "BC",
                          "creator_url": "http://www.bc.de"
                          },
                         flatten_dict(
                             {"title": "DCS4COP Sentinel-3 OLCI L2C Data Cube",
                              "creator": [
                                  {"name": "BC",
                                   "url": "http://www.bc.de"
                                   }
                              ]}))

        self.assertEqual({"title": "DCS4COP Sentinel-3 OLCI L2C Data Cube",
                          "creator_name": "BC, ACME",
                          "creator_url": "http://www.bc.de, http://acme.com"
                          },
                         flatten_dict(
                             {"title": "DCS4COP Sentinel-3 OLCI L2C Data Cube",
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
        d = yaml.full_load(stream)
        d = flatten_dict(d['output_metadata'])
        self.assertEqual(17, len(d))
        self.assertEqual('DCS4COP Sentinel-3 OLCI L2C Data Cube',
                         d.get('title'))
        self.assertEqual(
            'Brockmann Consult GmbH, Royal Belgian Institute for Natural Sciences (RBINS)',
            d.get('creator_name'))
        self.assertEqual(
            'https://www.brockmann-consult.de, http://odnature.naturalsciences.be/remsem/',
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


class LoadConfigsTest(unittest.TestCase):

    def tearDown(self) -> None:
        fs: fsspec.AbstractFileSystem = fsspec.filesystem("memory")
        fs.rm("/", recursive=True)

    def test_load_configs_ok(self):
        with fsspec.open("memory://config-1.yaml", mode="w") as fp:
            yaml.safe_dump(dict(a=1), fp)
        with fsspec.open("memory://config-2.yaml", mode="w") as fp:
            yaml.safe_dump(dict(b=2), fp)
        with fsspec.open("memory://config-3.yaml", mode="w") as fp:
            yaml.safe_dump(dict(c=3), fp)
        config = load_configs("memory://config-1.yaml",
                              "memory://config-2.yaml",
                              "memory://config-3.yaml")
        self.assertEqual({'a': 1, 'b': 2, 'c': 3}, config)

    # noinspection PyMethodMayBeStatic
    def test_load_configs_fails(self):
        with pytest.raises(ValueError,
                           match="Cannot find configuration"
                                 " 'memory://config_1.yaml'"):
            load_configs("memory://config_1.yaml")
