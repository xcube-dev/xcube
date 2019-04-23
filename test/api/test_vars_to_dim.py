import unittest
import xarray as xr

from test.sampledata import new_test_dataset
from xcube.api.vars_to_dim import vars_to_dim


class VarsToDimTest(unittest.TestCase):
    def test_vars_to_dim(self):
        dataset = new_test_dataset(["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-04", "2010-01-05"],
                                   precipitation=0.4, temperature=275.2)

        print(dataset.dims)

        ds = vars_to_dim(dataset)

        self.assertIn("newdim_vars", ds.variables)
        newdim_vars = ds["newdim_vars"]
        self.assertTrue(hasattr(newdim_vars, "encoding"))
        self.assertEqual(len(dataset.dims), 3)

        self.assertRaises(ValueError, vars_to_dim, xr.Dataset())
