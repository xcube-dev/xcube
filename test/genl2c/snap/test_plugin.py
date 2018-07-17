import unittest

from xcube.genl2c.snap.inputprocessor import init_plugin


class SnapPluginTest(unittest.TestCase):

    # noinspection PyMethodMayBeStatic
    def test_init_plugin(self):
        # Smoke test
        init_plugin()
