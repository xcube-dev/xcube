import os
import os.path
from typing import Mapping

from test.cli.helpers import CliDataTest, CliTest
from xcube.cli.serve import BASE_ENV_VAR
from xcube.cli.serve import CONFIG_ENV_VAR
from xcube.cli.serve import VIEWER_ENV_VAR
from xcube.cli.serve import main


class ServerCliTest(CliTest):
    def test_help(self):
        result = self.invoke_cli(["serve2", "--help"])
        self.assertEqual(0, result.exit_code)

    def test_stop_after(self):
        result = self.invoke_cli(["serve2", "--stop-after",  "0.1"])
        self.assertEqual(0, result.exit_code)
