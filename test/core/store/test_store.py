import unittest

from xcube.core.store.store import find_data_store_extensions
from xcube.core.store.store import get_data_store_params_schema
from xcube.util.jsonschema import JsonObjectSchema


class ExtensionRegistryTest(unittest.TestCase):

    def test_find_data_store_extensions(self):
        extensions = find_data_store_extensions()
        actual_ext = set(ext.name for ext in extensions)
        self.assertIn('mem', actual_ext)
        self.assertIn('dir', actual_ext)

    def test_get_data_store_params_schema(self):
        schema = get_data_store_params_schema('mem')
        self.assertIsInstance(schema, JsonObjectSchema)

        schema = get_data_store_params_schema('dir')
        self.assertIsInstance(schema, JsonObjectSchema)