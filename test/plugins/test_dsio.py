import unittest

from xcube.util.extension import ExtensionRegistry


class DatasetIOPluginTest(unittest.TestCase):
    def test_init_plugin(self):
        from xcube.plugins.dsio import init_plugin
        ext_reg = ExtensionRegistry()
        init_plugin(ext_reg)
        self.assertTrue(ext_reg.has_extension('xcube.core.dsio', 'zarr'))
        self.assertTrue(ext_reg.has_extension('xcube.core.dsio', 'netcdf4'))
        self.assertTrue(ext_reg.has_extension('xcube.core.dsio', 'csv'))
        self.assertTrue(ext_reg.has_extension('xcube.core.dsio', 'mem'))
