import unittest
from typing import List
from abc import ABCMeta
import click
import click.testing

from xcube.cli.gen import generate_cube as cli
# TODO AliceBalfanz: extended help section, reason for keeping code for GenL2CHelpFormatter
# from xcube.cli.gen import main, GenL2CHelpFormatter


class CliTest(unittest.TestCase, metaclass=ABCMeta):

    @classmethod
    def invoke_cli(cls, args: List[str]):
        runner = click.testing.CliRunner()
        return runner.invoke(cli, args)

    def test_help_option(self):
        result = self.invoke_cli(['time_irregular', '--help'])
        self.assertEqual(0, result.exit_code)

    def test_missing_args(self):
        result = self.invoke_cli([])
        # exit_code==0 when run from PyCharm, exit_code==2 else
        self.assertEqual(1, result.exit_code)

    def test_main_with_illegal_size_option(self):
        result = self.invoke_cli(['-s', '120,abc', 'input.nc'])
        self.assertEqual(1, result.exit_code)

    def test_main_with_illegal_region_option(self):
        result = self.invoke_cli(['-r', '50,_2,55,21', 'input.nc'])
        self.assertEqual(1, result.exit_code)
        result = self.invoke_cli(['-r', '50,20,55', 'input.nc'])
        self.assertEqual(1, result.exit_code)

    def test_main_with_illegal_variable_option(self):
        result = self.invoke_cli(['-v', ' ', 'input.nc'])
        self.assertEqual(1, result.exit_code)

    def test_main_with_illegal_config_option(self):
        result = self.invoke_cli(['-c', 'nonono.yml', 'input.nc'])
        self.assertEqual(1, result.exit_code)

    def test_main_with_illegal_options(self):
        result = self.invoke_cli(['input.nc'])
        self.assertEqual(1, result.exit_code)


# class GenL2CHelpFormatterTest(unittest.TestCase):
#     def test_custom_help(self):
#         help_text = GenL2CHelpFormatter('xcube').format_help()
#         self.assertIn(' netcdf4 ', help_text)
#         self.assertIn(' zarr ', help_text)
