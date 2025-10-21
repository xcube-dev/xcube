# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from test.cli.helpers import TEST_NC_FILE, CliDataTest

import xarray as xr


class DumpTest(CliDataTest):
    def test_dump_ds(self):
        self.maxDiff = None
        with xr.set_options(display_width=80):
            result = self.invoke_cli(["dump", TEST_NC_FILE])

        self.assertIn("<xarray.Dataset>", result.output)
        self.assertIn("Dimensions:        (lon: 360, lat: 180, time: 5, bnds: 2)\n", result.output)
        self.assertIn("Coordinates:\n", result.output)
        self.assertIn("  * lon            (lon) float64 ", result.output)
        self.assertIn("Data variables:\n", result.output)
        self.assertIn("    precipitation  (time, lat, lon) float64 ", result.output)
        self.assertIn("Attributes:\n", result.output)
        self.assertIn("title:                 Test Cube", result.output)

        variables_regex = r"""
Data variables:
    precipitation  \(time, lat, lon\) float64 (3MB)? ...
    temperature    \(time, lat, lon\) float64 (3MB)? ...
    soil_moisture  \(time, lat, lon\) float64 (3MB)? ...
"""
        
        self.assertRegex(result.output, variables_regex)
        self.assertEqual(0, result.exit_code)
