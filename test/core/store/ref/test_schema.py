# The MIT License (MIT)
# Copyright (c) 2020-2024 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
                'asynchronous',
                'cache_size',
                'listings_expiry_time',
                'max_block',
                'max_gap',
                'max_paths',
                'refs',
                'remote_options',
                'remote_protocol',
                'skip_instance_cache',
                'target_options',
                'target_protocol',
                'use_listings_cache'
            },
            set(REF_STORE_SCHEMA.properties.keys())
        )

    def test_json_serialisation(self):
        d = REF_STORE_SCHEMA.to_dict()
        self.assertEqual(d, json.loads(json.dumps(d)))
