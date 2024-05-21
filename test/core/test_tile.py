# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

import numpy as np
import xarray as xr

# noinspection PyProtectedMember
from xcube.core.tile import get_var_valid_range


class GetVarValidRangeTest(unittest.TestCase):
    def test_from_valid_range(self):
        a = xr.DataArray(0, attrs=dict(valid_range=[-1, 1]))
        self.assertEqual((-1, 1), get_var_valid_range(a))

    def test_from_valid_min_max(self):
        a = xr.DataArray(0, attrs=dict(valid_min=-1, valid_max=1))
        self.assertEqual((-1, 1), get_var_valid_range(a))

    def test_from_valid_min(self):
        a = xr.DataArray(0, attrs=dict(valid_min=-1))
        self.assertEqual((-1, np.inf), get_var_valid_range(a))

    def test_from_valid_max(self):
        a = xr.DataArray(0, attrs=dict(valid_max=1))
        self.assertEqual((-np.inf, 1), get_var_valid_range(a))

    def test_from_nothing(self):
        a = xr.DataArray(0)
        self.assertEqual(None, get_var_valid_range(a))
