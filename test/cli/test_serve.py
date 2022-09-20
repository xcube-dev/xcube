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
        result = self.invoke_cli(["serve", "list", "apis"])
        self.assertEqual(0, result.exit_code)

        result = self.invoke_cli(["serve", "show", "openapi"])
        self.assertEqual(0, result.exit_code)

        result = self.invoke_cli(["serve", "show", "config"])
        self.assertEqual(0, result.exit_code)

        result = self.invoke_cli(["serve", "show", "configschema"])
        self.assertEqual(0, result.exit_code)

        result = self.invoke_cli(["serve", "list", "routes"])
        self.assertEqual(1, result.exit_code)

    def test_commands_with_format(self):
        for f in ["yaml", "json"]:
            result = self.invoke_cli(["serve", "show", "openapi"] + [f])
            self.assertEqual(0, result.exit_code)

            result = self.invoke_cli(["serve", "show", "config"] + [f])
            self.assertEqual(0, result.exit_code)

            result = self.invoke_cli(["serve", "show", "configschema"] + [f])
            self.assertEqual(0, result.exit_code)

        result = self.invoke_cli(["serve", "show", "configschema", "nc"])
        self.assertEqual(1, result.exit_code)
