# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest
import json
from xcube.core.store.ref.schema import REF_STORE_SCHEMA
from xcube.util.jsonschema import JsonObjectSchema


class ReferenceSchemaTest(unittest.TestCase):
    def test_schema_schema(self):
        self.assertIsInstance(REF_STORE_SCHEMA, JsonObjectSchema)
        self.assertIsInstance(REF_STORE_SCHEMA.properties, dict)
        self.assertEqual(
            {
                "asynchronous",
                "cache_size",
                "listings_expiry_time",
                "max_block",
                "max_gap",
                "max_paths",
                "refs",
                "remote_options",
                "remote_protocol",
                "skip_instance_cache",
                "target_options",
                "target_protocol",
                "use_listings_cache",
            },
            set(REF_STORE_SCHEMA.properties.keys()),
        )

    def test_json_serialisation(self):
        d = REF_STORE_SCHEMA.to_dict()
        self.assertEqual(d, json.loads(json.dumps(d)))
