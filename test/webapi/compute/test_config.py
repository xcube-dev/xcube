# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from unittest import TestCase

from xcube.util.jsonschema import JsonObjectSchema
from xcube.webapi.compute.config import CONFIG_SCHEMA


class ComputeConfigTest(TestCase):
    def test_config_schema(self):
        self.assertIsInstance(CONFIG_SCHEMA, JsonObjectSchema)
        self.assertEqual(
            {
                "type": "object",
                "properties": {
                    "Compute": {
                        "type": "object",
                        "properties": {
                            "MaxWorkers": {
                                "type": "integer",
                                "minimum": 1,
                            }
                        },
                        "additionalProperties": False,
                    }
                },
            },
            CONFIG_SCHEMA.to_dict(),
        )
