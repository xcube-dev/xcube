import unittest

from xcube.util.jsonschema import JsonObjectSchema
from xcube.webapi.config import ServiceConfig


class ServiceConfigTest(unittest.TestCase):
    def test_get_schema(self):
        schema = ServiceConfig.get_schema()
        self.assertIsInstance(schema, JsonObjectSchema)
