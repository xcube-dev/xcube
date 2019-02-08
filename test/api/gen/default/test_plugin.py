import unittest

from xcube.api.gen.default.iproc import init_plugin


class DefaultPluginTest(unittest.TestCase):

    # noinspection PyMethodMayBeStatic
    def test_init_plugin(self):
        # Smoke test
        init_plugin()
