from test.cli.helpers import CliTest


class GenCliTest(CliTest):

    def test_help_option(self):
        result = self.invoke_cli(['gen', 'time_irregular', '--help'])
        self.assertEqual(0, result.exit_code)

    def test_help(self):
        result = self.invoke_cli(['gen', '--help'])
        self.assertEqual(0, result.exit_code)

    def test_info(self):
        result = self.invoke_cli(['gen', '--info'])
        self.assertEqual(0, result.exit_code)

    def test_missing_args(self):
        with self.assertRaises(ValueError) as cm:
            self.invoke_cli(['gen'])
        self.assertEqual("Missing input_processor", f'{cm.exception}')

    def test_main_with_illegal_size_option(self):
        with self.assertRaises(ValueError) as cm:
            self.invoke_cli(['gen', '-s', '120,abc', 'input.nc'])
        self.assertEqual("output_size must have the form <width>,<height>,"
                         " where both values must be positive integer numbers", f'{cm.exception}')

    def test_main_with_illegal_region_option(self):
        with self.assertRaises(ValueError) as cm:
            self.invoke_cli(['gen', '-r', '50,_2,55,21', 'input.nc'])
        self.assertEqual("output_region must have the form <lon_min>,<lat_min>,<lon_max>,<lat_max>,"
                         " where all four numbers must be floating point numbers in degrees", f'{cm.exception}')
        with self.assertRaises(ValueError) as cm:
            self.invoke_cli(['gen', '-r', '50,20,55', 'input.nc'])
        self.assertEqual("output_region must have the form <lon_min>,<lat_min>,<lon_max>,<lat_max>,"
                         " where all four numbers must be floating point numbers in degrees", f'{cm.exception}')

    def test_main_with_illegal_variable_option(self):
        with self.assertRaises(ValueError) as cm:
            self.invoke_cli(['gen', '-v', ' ', 'input.nc'])
        self.assertEqual("output_variables must be a list of existing variable names", f'{cm.exception}')

    def test_main_with_illegal_config_option(self):
        with self.assertRaises(ValueError) as cm:
            self.invoke_cli(['gen', '-c', 'nonono.yml', 'input.nc'])
        self.assertEqual("Cannot find configuration 'nonono.yml'", f'{cm.exception}')

    def test_main_with_illegal_options(self):
        with self.assertRaises(ValueError) as cm:
            self.invoke_cli(['gen', 'input.nc'])
        self.assertEqual('Missing input_processor', f'{cm.exception}')

