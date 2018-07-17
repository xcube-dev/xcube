import unittest
from io import StringIO

import yaml

from xcube.config import flatten_dict, to_name_dict_pair, to_name_dict_pairs, to_resolved_name_dict_pairs


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
