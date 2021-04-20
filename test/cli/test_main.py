import warnings

import pytest
from click import command

import xcube.cli.main
from test.cli.helpers import CliTest


class ClickMainTest(CliTest):
    USER_WARNING_MESSAGE = 'This is a user warning.'
    DEPRECATION_WARNING_MESSAGE = 'This is a deprecation warning.'
    RUNTIME_WARNING_MESSAGE = 'This is a runtime warning.'

    def setUp(self):
        @command(name='test_warnings')
        def test_warnings():
            warnings.warn(self.USER_WARNING_MESSAGE)
            warnings.warn(self.DEPRECATION_WARNING_MESSAGE, category=DeprecationWarning)
            warnings.warn(self.RUNTIME_WARNING_MESSAGE, category=RuntimeWarning)

        xcube.cli.main.cli.add_command(test_warnings)

    def test_warnings_not_issued_by_default(self):
        with pytest.warns(None) as record:
            self.invoke_cli(['test_warnings'])
        self.assertEqual(['This is a user warning.'],
                         list(map(lambda r: str(r.message), record.list)))

    def test_warnings_issued_when_enabled(self):
        with pytest.warns(None) as record:
            self.invoke_cli(['--warnings', 'test_warnings'])
        self.assertEqual(['This is a user warning.',
                          'This is a deprecation warning.',
                          'This is a runtime warning.'],
                         list(map(lambda r: str(r.message), record.list)))
