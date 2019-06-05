import unittest

from xcube.cli.agg import agg


class CliTest(unittest.TestCase):
    def test_help(self):
        exit_code = agg(['-h'])
        self.assertEqual(0, exit_code)
        exit_code = agg(['--help'])
        self.assertEqual(0, exit_code)

    def test_missing_args(self):
        exit_code = agg([])
        self.assertEqual(2, exit_code)

    def test_variable_option(self):
        exit_code = agg(['-v', ' ', 'input.nc'])
        self.assertEqual(2, exit_code)

    def test_metadata_option(self):
        exit_code = agg(['-m', 'nonono.yml', 'input.nc'])
        self.assertEqual(2, exit_code)

    def test_resampling_option(self):
        exit_code = agg(['--resampling', 'NeArEsT'])
        self.assertEqual(2, exit_code)

    def test_illegal_input(self):
        exit_code = agg(['input.nc'])
        self.assertEqual(2, exit_code)

