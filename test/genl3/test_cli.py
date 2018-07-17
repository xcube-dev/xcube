import unittest

from xcube.genl3.cli import main


class CliTest(unittest.TestCase):
    def test_help(self):
        exit_code = main(['-h'])
        self.assertEqual(0, exit_code)
        exit_code = main(['--help'])
        self.assertEqual(0, exit_code)

    def test_missing_args(self):
        exit_code = main([])
        self.assertEqual(2, exit_code)

    def test_variable_option(self):
        exit_code = main(['-v', ' ', 'input.nc'])
        self.assertEqual(2, exit_code)

    def test_metadata_option(self):
        exit_code = main(['-m', 'nonono.yml', 'input.nc'])
        self.assertEqual(2, exit_code)

    def test_resampling_option(self):
        exit_code = main(['--resampling', 'NeArEsT'])
        self.assertEqual(2, exit_code)

    def test_illegal_input(self):
        exit_code = main(['input.nc'])
        self.assertEqual(2, exit_code)

