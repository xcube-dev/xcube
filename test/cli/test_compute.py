# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import os
from test.cli.helpers import TEST_ZARR_DIR, CliDataTest
from typing import List

import xarray as xr

OUTPUT_PATH = "out.zarr"


class ComputeCliTest(CliDataTest):
    def outputs(self) -> list[str]:
        return [OUTPUT_PATH]

    def test_help_option(self):
        result = self.invoke_cli(["compute", "--help"])
        self.assertEqual(0, result.exit_code)

    def test_compute_with_init_and_finalize(self):
        result = self.invoke_cli(
            [
                "compute",
                os.path.join(
                    os.path.dirname(__file__),
                    "compute-scripts",
                    "with-init-and-finalize.py",
                ),
                TEST_ZARR_DIR,
            ]
        )
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir(OUTPUT_PATH))

        ds = xr.open_zarr(OUTPUT_PATH)
        self.assertEqual(["output"], list(ds.data_vars))
        self.assertEqual("mg/m^3", ds.output.attrs.get("units"))
        self.assertEqual("I has a bucket", ds.attrs.get("comment"))
        self.assertAlmostEqual(0.2 * 0.4 + 0.3 * 0.5, float(ds.output.mean()))

    def test_compute_without_init(self):
        result = self.invoke_cli(
            [
                "compute",
                "--params",
                "a=0.1,b=0.4",
                "--vars",
                "precipitation,soil_moisture",
                os.path.join(
                    os.path.dirname(__file__), "compute-scripts", "without-init.py"
                ),
                TEST_ZARR_DIR,
            ]
        )
        self.assertEqual(0, result.exit_code)
        self.assertTrue(os.path.isdir(OUTPUT_PATH))

        ds = xr.open_zarr(OUTPUT_PATH)
        self.assertEqual(["output"], list(ds.data_vars))
        self.assertAlmostEqual(0.1 * 0.4 + 0.4 * 0.5, float(ds.output.mean()))
