# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.


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
        operations: list = result["operations"]

        operations_map = {op.get("operationId"): op for op in operations}
        self.assertIn("spatial_subset", operations_map)

        op = operations_map["spatial_subset"]
        self.assert_spatial_subset_op_ok(op)

    def test_get_compute_operation_entry(self):
        op = get_compute_operation(get_compute_ctx(), "spatial_subset")
        self.assert_spatial_subset_op_ok(op)

    def assert_spatial_subset_op_ok(self, op: Any):
        self.assertIsInstance(op, dict)

        self.assertEqual("spatial_subset", op.get("operationId"))
        self.assertEqual(
            "Create a spatial subset" " from given dataset.", op.get("description")
        )
        self.assertIn("parametersSchema", op)

        schema = op.get("parametersSchema")
        self.assertIsInstance(schema, dict)

        self.assertEqual("object", schema.get("type"))
        self.assertEqual(False, schema.get("additionalProperties"))
        self.assertEqual({"dataset", "bbox"}, set(schema.get("required", [])))
        self.assertEqual(
            {
                "dataset": {
                    "type": "string",
                    "title": "Dataset identifier",
                },
                "bbox": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "prefixItems": [
                        {"type": "number"},
                        {"type": "number"},
                        {"type": "number"},
                        {"type": "number"},
                    ],
                    "title": "Bounding box",
                    "description": "Bounding box using the "
                    "dataset's CRS "
                    "coordinates",
                },
            },
            schema.get("properties"),
        )
