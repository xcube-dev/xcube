import unittest

from xcube.core.store.store import find_data_store_extensions


class ExtensionRegistryTest(unittest.TestCase):

    def test_find_data_store_extensions(self):
        extensions = find_data_store_extensions()
        actual_ext = set(ext.name for ext in extensions)
        self.assertIn('mem', actual_ext)
        self.assertIn('dir', actual_ext)
