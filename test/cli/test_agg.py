import unittest

from xcube.cli.agg import aggregate


class CliTest(unittest.TestCase):
    def test_help(self):
        exit_code = aggregate(['-h'])
        self.assertEqual(0, exit_code)
        exit_code = aggregate(['--help'])
        self.assertEqual(0, exit_code)

    def test_missing_args(self):
        exit_code = aggregate([])
        self.assertEqual(2, exit_code)

    def test_variable_option(self):
        exit_code = aggregate(['-v', ' ', 'input.nc'])
        self.assertEqual(2, exit_code)

    def test_metadata_option(self):
        exit_code = aggregate(['-m', 'nonono.yml', 'input.nc'])
        self.assertEqual(2, exit_code)

    def test_resampling_option(self):
        exit_code = aggregate(['--resampling', 'NeArEsT'])
        self.assertEqual(2, exit_code)

    def test_illegal_input(self):
        exit_code = aggregate(['input.nc'])
        self.assertEqual(2, exit_code)

