from test.cli.helpers import CliDataTest, CliTest


class GenCliTest(CliTest):

    def test_help_option(self):
        result = self.invoke_cli(['gen', 'time_irregular', '--help'])
        self.assertEqual(0, result.exit_code)

    # TODO (Alicja): reactive following tests
    # # This test fails, although an error is thrown - and then the click packages changes it into 0.
    # def test_missing_args(self):
    #     result = self.invoke_cli([])
    #     # exit_code==0 when run from PyCharm, exit_code==2 else
    #     self.assertEqual(1, result.exit_code)
    #
    # # This test fails, although an error is thrown - and then the click packages changes it into 0.
    # def test_main_with_illegal_size_option(self):
    #     result = self.invoke_cli(['-s', '120,abc', 'input.nc'])
    #     self.assertEqual(1, result.exit_code)
    #
    # # This test fails, although an error is thrown - and then the click packages changes it into 0.
    # def test_main_with_illegal_region_option(self):
    #     result = self.invoke_cli(['-r', '50,_2,55,21', 'input.nc'])
    #     self.assertEqual(1, result.exit_code)
    #     result = self.invoke_cli(['-r', '50,20,55', 'input.nc'])
    #     self.assertEqual(1, result.exit_code)
    #
    # # This test fails, although an error is thrown - and then the click packages changes it into 0.
    # def test_main_with_illegal_variable_option(self):
    #     result = self.invoke_cli(['-v', ' ', 'input.nc'])
    #     self.assertEqual(1, result.exit_code)
    #
    # def test_main_with_illegal_config_option(self):
    #     result = self.invoke_cli(['-c', 'nonono.yml', 'input.nc'])
    #     self.assertEqual(1, result.exit_code)
    #
    # # This test fails, although an error is thrown - and then the click packages changes it into 0.
    # def test_main_with_illegal_options(self):
    #     result = self.invoke_cli(['input.nc'])
    #     self.assertEqual(1, result.exit_code)

    def test_info_true(self):
        result = self.invoke_cli(['gen', '--info'])
        self.assertEqual(0, result.exit_code)
