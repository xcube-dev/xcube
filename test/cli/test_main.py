from test.cli.helpers import CliTest
from click import command
import xcube.cli.main
from warnings import warn
import pytest


class ClickMainTest(CliTest):

    WARNING_MESSAGE = 'This is a test warning.'

    def setUp(self):
        @command(name='test_warnings')
        def test_warnings():
            warn(self.WARNING_MESSAGE)

        xcube.cli.main.cli.add_command(test_warnings)

    def test_warnings_not_issued_by_default(self):
        with pytest.warns(None) as record:
            self.invoke_cli(['test_warnings'])
        self.assertEqual(0, len(record.list))

    def test_warnings_issued_when_enabled(self):
        with pytest.warns(UserWarning, match=self.WARNING_MESSAGE):
            self.invoke_cli(['--warnings', 'test_warnings'])
