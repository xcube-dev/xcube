# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

from xcube.core.new import new_cube
from xcube.core.vars2dim import vars_to_dim


class VarsToDimTest(unittest.TestCase):
    def test_vars_to_dim(self):
        dataset = new_cube(variables=dict(precipitation=0.4, temperature=275.2))

        ds = vars_to_dim(dataset)

        self.assertIn("var", ds.sizes)
        self.assertEqual(2, ds.sizes["var"])
        self.assertIn("var", ds.coords)
        self.assertIn("data", ds.data_vars)
        var_names = ds["var"]
        self.assertEqual(("var",), var_names.dims)
        self.assertTrue(hasattr(var_names, "encoding"))
        self.assertEqual(2, len(var_names))
        self.assertIn("precipitation", str(var_names[0]))
        self.assertIn("temperature", str(var_names[1]))
