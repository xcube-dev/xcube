import unittest

from xcube.util.ext import ExtensionRegistry


class CliCommandPluginTest(unittest.TestCase):
    def test_init_plugin(self):
        from xcube.plugins.cli import init_plugin
        ext_reg = ExtensionRegistry()
        init_plugin(ext_reg)
        # self.assertTrue(ext_reg.has_ext('cli', 'optimize'))
