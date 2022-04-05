import os
import os.path
from typing import Mapping

from test.cli.helpers import CliDataTest
from xcube.cli.serve import BASE_ENV_VAR
from xcube.cli.serve import CONFIG_ENV_VAR
from xcube.cli.serve import VIEWER_ENV_VAR
from xcube.cli.serve import main


class ServerCliTest(CliDataTest):
    def test_config(self):
        result = self.invoke_cli(["serve", "--config", "x.yml", "pippo.zarr"])
        self.assertEqual('Error: CONFIG and CUBES cannot be used at the same time.\n',
                         result.stderr[-58:])
        self.assertEqual(1, result.exit_code)

        with ChangeEnv({CONFIG_ENV_VAR: "x.yml"}):
            result = self.invoke_cli(["serve"])
            self.assertEqual('Error: Configuration file not found: x.yml\n',
                             result.stderr[-44:])
            self.assertEqual(1, result.exit_code)

    def test_base_dir(self):
        config_path = os.path.join(os.path.dirname(__file__),
                                   '..', '..', 'examples', 'serve', 'demo', 'config.yml')

        with ChangeEnv({BASE_ENV_VAR: None}):
            result = self.invoke_cli(["serve", "--config", config_path, "--base-dir", "pippo/gnarz"])
            self.assertEqual('Error: Base directory not found: pippo/gnarz\n',
                             result.stderr[-45:])
            self.assertEqual(1, result.exit_code)

        with ChangeEnv({BASE_ENV_VAR: 'pippo/configs'}):
            result = self.invoke_cli(["serve", "--config", config_path])
            self.assertEqual('Error: Base directory not found: pippo/configs\n',
                             result.stderr[-47:])
            self.assertEqual(1, result.exit_code)

    def test_show(self):
        config_path = os.path.join(os.path.dirname(__file__),
                                   '..', '..', 'examples', 'serve', 'demo', 'config.yml')

        with ChangeEnv({VIEWER_ENV_VAR: None}):
            result = self.invoke_cli(["serve", "--show", "--config", config_path])
            self.assertEqual(f'Error: Option "--show": In order to run the viewer, '
                             f'set environment variable {VIEWER_ENV_VAR} to a valid '
                             f'xcube-viewer deployment or build directory.\n',
                             result.stderr[-150:])
            self.assertEqual(2, result.exit_code)

        with ChangeEnv({VIEWER_ENV_VAR: "pippo/viewer"}):
            result = self.invoke_cli(["serve", "--show", "--config", config_path])
            self.assertEqual(f'Error: Option "--show": Viewer path set by environment '
                             f'variable {VIEWER_ENV_VAR} '
                             f'must be a directory: pippo/viewer\n',
                             result.stderr[-116:])
            self.assertEqual(2, result.exit_code)

    def test_help(self):
        with self.assertRaises(SystemExit):
            main(args=["--help"])


class ChangeEnv:
    def __init__(self, env=None, **env_kwargs):
        self._new_env = dict(env) if env else dict()
        self._new_env.update(**env_kwargs)
        self._old_env = None

    def __enter__(self) -> Mapping[str, str]:
        self._old_env = dict()
        for k, v in self._new_env.items():
            self._old_env[k] = os.environ.get(k)
            if v is not None:
                os.environ[k] = str(v)
            elif k in os.environ:
                del os.environ[k]
        return os.environ

    def __exit__(self, *error):
        for k, v in self._old_env.items():
            if v is not None:
                os.environ[k] = v
            elif k in os.environ:
                del os.environ[k]
