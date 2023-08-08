# The MIT License (MIT)
# Copyright (c) 2023 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import unittest
from typing import List, Any

from xcube.webapi.compute.controllers import get_compute_operations
from xcube.webapi.compute.controllers import get_compute_operation
from .test_context import get_compute_ctx


class ComputeControllersTest(unittest.TestCase):
    def test_get_compute_operations(self):
        result = get_compute_operations(get_compute_ctx())
        self.assertIsInstance(result, dict)
        self.assertIn("operations", result)
        self.assertIsInstance(result["operations"], list)
        self.assertTrue(len(result["operations"]) > 0)

    def test_get_compute_operations_entry(self):
        result = get_compute_operations(get_compute_ctx())
        operations: List = result["operations"]

        operations_map = {op.get('operationId'): op for op in operations}
        self.assertIn('spatial_subset', operations_map)

        op = operations_map['spatial_subset']
        self.assert_spatial_subset_op_ok(op)

    def test_get_compute_operation_entry(self):
        op = get_compute_operation(get_compute_ctx(), 'spatial_subset')
        self.assert_spatial_subset_op_ok(op)

    def assert_spatial_subset_op_ok(self, op: Any):
        self.assertIsInstance(op, dict)

        self.assertEqual('spatial_subset', op.get('operationId'))
        self.assertEqual('Create a spatial subset'
                         ' from given dataset.',
                         op.get('description'))
        self.assertIn("parametersSchema", op)

        schema = op.get("parametersSchema")
        self.assertIsInstance(schema, dict)

        self.assertEqual('object',
                         schema.get('type'))
        self.assertEqual(False,
                         schema.get('additionalProperties'))
        self.assertEqual({'dataset', 'bbox'},
                         set(schema.get('required', [])))
        self.assertEqual(
            {
                'dataset': {
                    'type': 'string',
                    'title': 'Dataset identifier',
                },
                'bbox': {
                    'type': 'array',
                    'minItems': 4,
                    'maxItems': 4,
                    'prefixItems': [{'type': 'number'},
                              {'type': 'number'},
                              {'type': 'number'},
                              {'type': 'number'}],
                    'title': 'Bounding box',
                    'description': 'Bounding box using the '
                                   'dataset\'s CRS '
                                   'coordinates',
                },
            },
            schema.get('properties')
        )
