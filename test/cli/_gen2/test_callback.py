import unittest
from xcube.cli._gen2.request import Callback


class TestCallback(unittest.TestCase):
    def test_progress(self):
        with self.assertRaises(ValueError) as e:
            Callback()
        self.assertEqual('Both, api_uri and access_token must be given', str(e.exception))

        expected = {"api_uri": 'https://bla.com', "access_token": 'dfasovjdasoövjidfs'}
        callback = Callback(**expected)
        res = callback.to_dict()

        self.assertDictEqual(expected, res)




