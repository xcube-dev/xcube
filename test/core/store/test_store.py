import unittest

from xcube.core.store import DataStoreError
from xcube.core.store.store import find_data_store_extensions
from xcube.core.store.store import get_data_store_class
from xcube.core.store.store import get_data_store_params_schema
from xcube.util.jsonschema import JsonObjectSchema


class ExtensionRegistryTest(unittest.TestCase):

    def test_find_data_store_extensions(self):
        extensions = find_data_store_extensions()
        actual_ext = set(ext.name for ext in extensions)
        self.assertIn('memory', actual_ext)
        self.assertIn('directory', actual_ext)

    def test_get_data_store_params_schema(self):
        schema = get_data_store_params_schema('memory')
        self.assertIsInstance(schema, JsonObjectSchema)

        schema = get_data_store_params_schema('directory')
        self.assertIsInstance(schema, JsonObjectSchema)

    def test_get_data_store_class(self):
        cls = get_data_store_class('memory')
        self.assertIsInstance(cls, type)

        with self.assertRaises(DataStoreError) as cm:
            get_data_store_class('pippo')
        self.assertEqual('Unknown data store "pippo" (may be due to missing xcube plugin)',
                         f'{cm.exception}')
