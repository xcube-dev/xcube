import os

from test.cli.helpers import CliDataTest
from xcube.cli.serve import BASE_ENV_VAR
from xcube.cli.serve import CONFIG_ENV_VAR
from xcube.cli.serve import VIEWER_ENV_VAR
from xcube.cli.serve import main


class ServerCliTest(CliDataTest):
    def test_config(self):
        result = self.invoke_cli(["serve", "--config", "x.yml", "pippo.zarr"])
        self.assertEqual(1, result.exit_code)
        self.assertEqual('Error: CONFIG and CUBES cannot be used at the same time.\n',
                         result.output[-58:])

        old_value = os.environ.get(CONFIG_ENV_VAR)
        os.environ[CONFIG_ENV_VAR] = "x.yml"
        try:
            result = self.invoke_cli(["serve"])
            self.assertEqual(1, result.exit_code)
            self.assertEqual('Error: Configuration file not found: x.yml\n',
                             result.output[-44:])
        finally:
            if old_value is not None:
                os.environ[CONFIG_ENV_VAR] = old_value

    def test_base_dir(self):
        config_path = os.path.join(os.path.dirname(__file__),
                                   '..', '..', 'examples', 'serve', 'demo', 'config.yml')

        result = self.invoke_cli(["serve", "--config", config_path, "--base-dir", "pippo/gnarz"])
        self.assertEqual(1, result.exit_code)
        self.assertEqual('Error: Base directory not found: pippo/gnarz\n',
                         result.output[-45:])

        old_value = os.environ.get(BASE_ENV_VAR)
        os.environ[BASE_ENV_VAR] = 'pippo/curry'
        try:
            result = self.invoke_cli(["serve", "--config", config_path])
            self.assertEqual(1, result.exit_code)
            self.assertEqual('Error: Base directory not found: pippo/curry\n',
                             result.output[-45:])
        finally:
            if old_value is not None:
                os.environ[BASE_ENV_VAR] = old_value

    def test_show(self):
        config_path = os.path.join(os.path.dirname(__file__),
                                   '..', '..', 'examples', 'serve', 'demo', 'config.yml')

        viewer_path = os.environ.get(VIEWER_ENV_VAR)

        if VIEWER_ENV_VAR in os.environ:
            del os.environ[VIEWER_ENV_VAR]

        try:
            result = self.invoke_cli(["serve", "--show", "--config", config_path])
            self.assertEqual(2, result.exit_code)
            self.assertEqual(f'Error: Option "--show": In order to run the viewer, '
                             f'set environment variable {VIEWER_ENV_VAR} to a valid '
                             f'xcube-viewer deployment or build directory.\n',
                             result.output[-150:])

            os.environ[VIEWER_ENV_VAR] = "pip/po"
            result = self.invoke_cli(["serve", "--show", "--config", config_path])
            self.assertEqual(2, result.exit_code)
            self.assertEqual(f'Error: Option "--show": Viewer path set by environment '
                             f'variable {VIEWER_ENV_VAR} '
                             f'must be a directory: pip/po\n',
                             result.output[-110:])
        finally:
            if viewer_path is not None:
                os.environ[VIEWER_ENV_VAR] = viewer_path

    def test_help(self):
        with self.assertRaises(SystemExit):
            main(args=["--help"])
