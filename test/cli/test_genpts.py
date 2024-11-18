# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import os.path
from typing import List

import fiona
import pandas as pd

from test.cli.helpers import CliDataTest
from test.cli.helpers import TEST_ZARR_DIR

TEST_CSV = os.path.join(os.path.dirname(__file__), "out.csv")
TEST_GEOJSON = os.path.join(os.path.dirname(__file__), "out.geojson")


class GenptsCliTest(CliDataTest):
    def outputs(self) -> list[str]:
        return [TEST_CSV, TEST_GEOJSON]

    def test_help_option(self):
        result = self.invoke_cli(["extract", "--help"])
        self.assertEqual(0, result.exit_code)

    def test_csv(self):
        result = self.invoke_cli(["genpts", TEST_ZARR_DIR, "--output", TEST_CSV])
        self.assertEqual(0, result.exit_code)
        self.assertEqual("", result.stdout)
        df = pd.read_csv(TEST_CSV)
        self.assertEqual(100, len(df))

    def test_geojson(self):
        result = self.invoke_cli(["genpts", TEST_ZARR_DIR, "--output", TEST_GEOJSON])
        self.assertEqual(0, result.exit_code)
        self.assertEqual("", result.stdout)
        with fiona.open(TEST_GEOJSON) as collection:
            features = [feature for feature in collection]
        self.assertEqual(100, len(features))
