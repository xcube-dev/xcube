import unittest

from xcube.genl2c.default.inputprocessor import init_plugin


class DefaultPluginTest(unittest.TestCase):

    # noinspection PyMethodMayBeStatic
    def test_init_plugin(self):
        # Smoke test
        init_plugin()
