from test.cli.helpers import CliTest
from xcube.util.temp import new_temp_file


class VersionsTest(CliTest):

    def test_help_option(self):
        result = self.invoke_cli(['versions', '--help'])
        self.assertEqual(0, result.exit_code)

    def test_no_args(self):
        result = self.invoke_cli(['versions'])
        self.assertEqual(0, result.exit_code)
        self.assertYaml(result.stdout)

    def test_format(self):
        result = self.invoke_cli(['versions', '-f', 'json'])
        self.assertEqual(0, result.exit_code)
        self.assertJson(result.stdout)

        result = self.invoke_cli(['versions', '-f', 'yaml'])
        self.assertEqual(0, result.exit_code)
        self.assertYaml(result.stdout)

    def test_output(self):
        _, output_path = new_temp_file(suffix='.json')
        result = self.invoke_cli(['versions', '-o', output_path])
        self.assertEqual(0, result.exit_code)
        with open(output_path) as fp:
            text = fp.read()
        self.assertJson(text)

        _, output_path = new_temp_file(suffix='.yml')
        result = self.invoke_cli(['versions', '-o', output_path])
        self.assertEqual(0, result.exit_code)
        with open(output_path) as fp:
            text = fp.read()
        self.assertYaml(text)

    def assertJson(self, json_result: str):
        self.assertIn('{', json_result)
        self.assertIn(':', json_result)
        self.assertIn('}', json_result)

    def assertYaml(self, json_result: str):
        self.assertNotIn('{', json_result)
        self.assertIn(':', json_result)
        self.assertNotIn('}', json_result)
