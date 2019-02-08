import unittest

from xcube.cli.gen import main, GenL2CHelpFormatter


class CliTest(unittest.TestCase):
    def test_main_with_option_help(self):
        exit_code = main(['-h'])
        self.assertEqual(0, exit_code)
        exit_code = main(['--help'])
        self.assertEqual(0, exit_code)

    def test_main_with_missing_args(self):
        exit_code = main([])
        # exit_code==0 when run from PyCharm, exit_code==2 else
        self.assertEqual(2, exit_code)
        #self.assertTrue(exit_code in [0, 2])

    def test_main_with_illegal_size_option(self):
        exit_code = main(['-s', '120,abc', 'input.nc'])
        self.assertEqual(2, exit_code)

    def test_main_with_illegal_region_option(self):
        exit_code = main(['-r', '50,_2,55,21', 'input.nc'])
        self.assertEqual(2, exit_code)
        exit_code = main(['-r', '50,20,55', 'input.nc'])
        self.assertEqual(2, exit_code)

    def test_main_with_illegal_variable_option(self):
        exit_code = main(['-v', ' ', 'input.nc'])
        self.assertEqual(2, exit_code)

    def test_main_with_illegal_config_option(self):
        exit_code = main(['-c', 'nonono.yml', 'input.nc'])
        self.assertEqual(2, exit_code)

    def test_main_with_illegal_options(self):
        exit_code = main(['input.nc'])
        self.assertEqual(2, exit_code)


class GenL2CHelpFormatterTest(unittest.TestCase):
    def test_custom_help(self):
        help_text = GenL2CHelpFormatter('xcube').format_help()
        self.assertIn(' netcdf4 ', help_text)
        self.assertIn(' zarr ', help_text)
