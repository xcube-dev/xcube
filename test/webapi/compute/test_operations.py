# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from unittest import TestCase

import xarray as xr

from xcube.core.new import new_cube
from xcube.webapi.compute.op.registry import OP_REGISTRY
from xcube.webapi.compute.operations import spatial_subset


class ComputeOperationsTest(TestCase):
    def test_operations_registered(self):
        ops = OP_REGISTRY.ops
        self.assertIn("spatial_subset", ops)
        self.assertTrue(callable(ops["spatial_subset"]))

    def test_spatial_subset(self):
        dataset = new_cube()
        cube_subset = spatial_subset(dataset, bbox=(0, 0, 10, 20))
        self.assertIsInstance(cube_subset, xr.Dataset)
