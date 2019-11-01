import unittest

from xcube.util.extension import ExtensionRegistry


class CliCommandPluginTest(unittest.TestCase):
    def test_init_plugin(self):
        from xcube.plugins.cli import init_plugin
        ext_reg = ExtensionRegistry()
        init_plugin(ext_reg)
        self.assertTrue(ext_reg.has_extension('xcube.cli', 'compute'))
        self.assertTrue(ext_reg.has_extension('xcube.cli', 'extract'))
        self.assertTrue(ext_reg.has_extension('xcube.cli', 'gen'))
        self.assertTrue(ext_reg.has_extension('xcube.cli', 'level'))
        self.assertTrue(ext_reg.has_extension('xcube.cli', 'optimize'))
        # ...
