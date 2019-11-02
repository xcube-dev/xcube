from test.cli.test_cli import CliDataTest

from xcube.cli.serve import VIEWER_ENV_VAR
from xcube.cli.serve import main


class ServerCliTest(CliDataTest):
    def test_config(self):
        result = self.invoke_cli(["serve", "--config", "x.yml", "pippo.zarr"])
        self.assertEqual(1, result.exit_code)
        self.assertEqual('Error: CONFIG and CUBES cannot be used at the same time.\n',
                         result.output[-58:])

        result = self.invoke_cli(["serve", "--config", "x.yml", "pippo.zarr"])
        self.assertEqual(1, result.exit_code)
        self.assertEqual('Error: CONFIG and CUBES cannot be used at the same time.\n',
                         result.output[-58:])

    def test_show(self):
        import os
        viewer_path = os.environ.get(VIEWER_ENV_VAR)

        if viewer_path is not None:
            del os.environ[VIEWER_ENV_VAR]

        result = self.invoke_cli(["serve", "--show"])
        self.assertEqual(2, result.exit_code)
        self.assertEqual(f'Error: Option "--show": In order to run the viewer, '
                         f'set environment variable {VIEWER_ENV_VAR} to a valid '
                         f'xcube-viewer deployment or build directory.\n',
                         result.output[-150:])

        os.environ[VIEWER_ENV_VAR] = "pip/po"
        result = self.invoke_cli(["serve", "--show"])
        self.assertEqual(2, result.exit_code)
        self.assertEqual(f'Error: Option "--show": Viewer path set by environment '
                         f'variable {VIEWER_ENV_VAR} '
                         f'must be a directory: pip/po\n',
                         result.output[-110:])

        if viewer_path is not None:
            os.environ[VIEWER_ENV_VAR] = viewer_path

    def test_help(self):
        with self.assertRaises(SystemExit):
            main(args=["--help"])
