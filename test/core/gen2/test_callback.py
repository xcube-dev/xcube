import unittest
from xcube.core.gen2.genconfig import CallbackConfig


class TestCallbackConfig(unittest.TestCase):
    def test_callback(self):
        with self.assertRaises(ValueError) as e:
            CallbackConfig()
        self.assertEqual('Both, api_uri and access_token must be given', str(e.exception))

        expected = {"api_uri": 'https://bla.com', "access_token": 'dfasovjdasoövjidfs'}
        callback = CallbackConfig(**expected)
        res = callback.to_dict()

        self.assertDictEqual(expected, res)
