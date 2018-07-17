import unittest

from xcube.genl2c.rbins.inputprocessor import init_plugin


class RbinsPluginTest(unittest.TestCase):

    # noinspection PyMethodMayBeStatic
    def test_init_plugin(self):
        # Smoke test
        init_plugin()
