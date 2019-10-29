import unittest
from abc import ABCMeta
from typing import List

import click
import click.testing

from xcube.api.new import new_cube
from xcube.cli.main import cli
from xcube.util.dsio import rimraf

TEST_NC_FILE = "test.nc"
TEST_ZARR_DIR = "test.zarr"


class CliTest(unittest.TestCase, metaclass=ABCMeta):

    def invoke_cli(self, args: List[str]):
        self.runner = click.testing.CliRunner()
        return self.runner.invoke(cli, args, catch_exceptions=False)


class CliDataTest(CliTest, metaclass=ABCMeta):

    def outputs(self) -> List[str]:
        return []

    def setUp(self):
        self._rm_outputs()
        dataset = new_cube(variables=dict(precipitation=0.4, temperature=275.2, soil_moisture=0.5))
        dataset.to_netcdf(TEST_NC_FILE, mode="w")
        dataset.to_zarr(TEST_ZARR_DIR, mode="w")

    def tearDown(self):
        self._rm_outputs()

    def _rm_outputs(self):
        for path in [TEST_ZARR_DIR, TEST_NC_FILE] + self.outputs():
            rimraf(path)
