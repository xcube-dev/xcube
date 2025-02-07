# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest
from abc import ABCMeta
from typing import Dict, List, Optional

import click
import click.testing

from xcube.cli.main import cli
from xcube.core.dsio import rimraf
from xcube.core.new import new_cube

TEST_NC_FILE = "test.nc"
TEST_ZARR_DIR = "test.zarr"


class CliTest(unittest.TestCase, metaclass=ABCMeta):
    def invoke_cli(self, args: list[str]):
        self.runner = click.testing.CliRunner(mix_stderr=False)
        # noinspection PyTypeChecker
        return self.runner.invoke(cli, args, catch_exceptions=False)


class CliDataTest(CliTest, metaclass=ABCMeta):
    def outputs(self) -> list[str]:
        return []

    def time_periods(self) -> int:
        return 5

    def chunks(self) -> Optional[dict[str, int]]:
        return None

    def setUp(self):
        self._rm_outputs()
        dataset = new_cube(
            variables=dict(precipitation=0.4, temperature=275.2, soil_moisture=0.5),
            time_periods=self.time_periods(),
        )
        if self.chunks() is not None:
            dataset = dataset.chunk(self.chunks())
        dataset.to_netcdf(TEST_NC_FILE, mode="w")
        dataset.to_zarr(TEST_ZARR_DIR, mode="w")

    def tearDown(self):
        self._rm_outputs()

    def _rm_outputs(self):
        for path in [TEST_ZARR_DIR, TEST_NC_FILE] + self.outputs():
            rimraf(path)
