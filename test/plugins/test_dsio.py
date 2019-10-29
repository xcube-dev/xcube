import unittest

from xcube.util.ext import ExtensionRegistry


class DatasetIOPluginTest(unittest.TestCase):
    def test_init_plugin(self):
        from xcube.plugins.dsio import init_plugin
        ext_reg = ExtensionRegistry()
        init_plugin(ext_reg)
        self.assertTrue(ext_reg.has_ext('dsio', 'zarr'))
        self.assertTrue(ext_reg.has_ext('dsio', 'netcdf4'))
        self.assertTrue(ext_reg.has_ext('dsio', 'csv'))
        self.assertTrue(ext_reg.has_ext('dsio', 'mem'))
