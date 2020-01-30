import os

import xarray as xr

from test.cli.helpers import CliDataTest, TEST_ZARR_DIR


class Vars2DimTest(CliDataTest):
    TEST_OUTPUT = "test-vars2dim.zarr"

    def outputs(self):
        return [self.TEST_OUTPUT]

    def test_vars2dim(self):
        result = self.invoke_cli(["vars2dim", TEST_ZARR_DIR])

        output_path = self.TEST_OUTPUT
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir(output_path))

        ds = xr.open_zarr(output_path)
        self.assertIn("var", ds.dims)
        self.assertEqual(3, ds.dims["var"])
        self.assertIn("var", ds.coords)
        self.assertIn("data", ds.data_vars)
        var_names = ds["var"]
        self.assertEqual(("var",), var_names.dims)
        self.assertTrue(hasattr(var_names, "encoding"))
        self.assertEqual(3, len(var_names))
        self.assertIn("precipitation", str(var_names[0]))
        self.assertIn("soil_moisture", str(var_names[1]))
        self.assertIn("temperature", str(var_names[2]))
