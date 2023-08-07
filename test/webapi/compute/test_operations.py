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
