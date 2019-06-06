import unittest

from xcube.util.caseless import caseless_dict


class CaselessDictTest(unittest.TestCase):
    def test_contains(self):
        d = caseless_dict({'TIME': '2010-04-09', 'LON': 53.6})
        self.assertIn('time', d)
        self.assertIn('TIME', d)
        self.assertIn('lon', d)
        self.assertIn('LON', d)
        self.assertNotIn('lat', d)
        self.assertNotIn('LAT', d)

    def test_getitem(self):
        d = caseless_dict({'TIME': '2010-04-09', 'LON': 53.6})
        self.assertEqual('2010-04-09', d['time'])
        self.assertEqual('2010-04-09', d['TIME'])
        self.assertEqual(53.6, d['lon'])
        self.assertEqual(53.6, d['LON'])

        with self.assertRaises(KeyError):
            # noinspection PyUnusedLocal
            var = d['lat']

    def test_setitem(self):
        d = caseless_dict()
        d['TIME'] = '2010-04-09'
        d['LON'] = 53.6
        self.assertIn('time', d)
        self.assertIn('TIME', d)
        self.assertEqual('2010-04-09', d['time'])
        self.assertIn('lon', d)
        self.assertIn('LON', d)
        self.assertEqual(53.6, d['lon'])
        self.assertNotIn('lat', d)
        self.assertNotIn('LAT', d)

    def test_delitem(self):
        d = caseless_dict({'TIME': '2010-04-09', 'LON': 53.6})
        del d['TIME']
        self.assertNotIn('time', d)
        self.assertNotIn('TIME', d)
        self.assertIn('lon', d)
        self.assertIn('LON', d)
        del d['lon']
        self.assertNotIn('lon', d)
        self.assertNotIn('LON', d)

    def test_get(self):
        d = caseless_dict({'TIME': '2010-04-09', 'LON': 53.6})
        self.assertEqual('2010-04-09', d.get('time'))
        self.assertEqual('2010-04-09', d.get('TIME'))
        self.assertEqual(53.6, d.get('lon'))
        self.assertEqual(53.6, d.get('LON'))
        self.assertEqual(None, d.get('lat'))
        self.assertEqual(-3.1, d.get('LAT', -3.1))

    def test_pop(self):
        d = caseless_dict({'TIME': '2010-04-09', 'LON': 53.6})
        v = d.pop('TIME')
        self.assertEqual('2010-04-09', v)
        self.assertNotIn('time', d)
        self.assertNotIn('TIME', d)
        self.assertIn('lon', d)
        self.assertIn('LON', d)
        v = d.pop('lon')
        self.assertEqual(53.6, v)
        self.assertNotIn('lon', d)
        self.assertNotIn('LON', d)
