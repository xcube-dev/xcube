from test.cli.helpers import CliTest


class ServerCliTest(CliTest):
    def test_help(self):
        result = self.invoke_cli(["serve", "--help"])
        self.assertEqual(0, result.exit_code)

    def test_update_after(self):
        result = self.invoke_cli(["serve",
                                  "--update-after", "0.1",
                                  "--stop-after", "0.2"])
        self.assertEqual(0, result.exit_code)

    def test_commands(self):
        result = self.invoke_cli(["serve", "--show", "apis"])
        self.assertEqual(0, result.exit_code)

        result = self.invoke_cli(["serve", "--show", "endpoints"])
        self.assertEqual(0, result.exit_code)

        result = self.invoke_cli(["serve", "--show", "openapi"])
        self.assertEqual(0, result.exit_code)

        result = self.invoke_cli(["serve", "--show", "config"])
        self.assertEqual(0, result.exit_code)

        result = self.invoke_cli(["serve", "--show", "configschema"])
        self.assertEqual(0, result.exit_code)

        result = self.invoke_cli(["serve", "--show", "routes"])
        self.assertEqual(2, result.exit_code)

    def test_commands_with_format(self):
        for f in ["yaml", "json"]:
            result = self.invoke_cli(["serve", "--show", f"apis.{f}"])
            self.assertEqual(0, result.exit_code)

            result = self.invoke_cli(["serve", "--show", f"endpoints.{f}"])
            self.assertEqual(0, result.exit_code)

            result = self.invoke_cli(["serve", "--show", f"openapi.{f}"])
            self.assertEqual(0, result.exit_code)

            result = self.invoke_cli(["serve", "--show", f"config.{f}"])
            self.assertEqual(0, result.exit_code)

            result = self.invoke_cli(["serve", "--show", f"configschema.{f}"])
            self.assertEqual(0, result.exit_code)

        result = self.invoke_cli(["serve", "--show", "configschema.nc"])
        self.assertEqual(2, result.exit_code)
