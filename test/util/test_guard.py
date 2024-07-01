# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

import pytest
import numpy as np
import xarray as xr

from xcube.util.guard import new_type_guard


class GuardTest(unittest.TestCase):
    def test_get_attrib(self):

        # noinspection PyPep8Naming
        DataArrayGuard = new_type_guard(xr.DataArray, attrs={"dims", "sizes"})

        da = xr.DataArray([1, 2, 3], dims="x")
        dag = DataArrayGuard(da)

        self.assertEqual("DataArrayGuard", DataArrayGuard.__name__)
        self.assertEqual(("x",), da.dims)
        self.assertEqual(("x",), dag.dims)
        self.assertEqual({"x": 3}, da.sizes)
        self.assertEqual({"x": 3}, dag.sizes)
        self.assertEqual([1, 2, 3], list(da.values))
        with pytest.raises(
            AttributeError,
            match="attribute 'values' of 'DataArray' object is protected",
        ):
            # noinspection PyUnusedLocal
            result = dag.values
