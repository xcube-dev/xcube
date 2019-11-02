import unittest

import click

from xcube.cli.common import parse_cli_kwargs, handle_cli_exception


class ClickUtilTest(unittest.TestCase):

    def test_parse_cli_kwargs(self):
        self.assertEqual(dict(),
                         parse_cli_kwargs("", metavar="<chunks>"))
        self.assertEqual(dict(time=1, lat=256, lon=512),
                         parse_cli_kwargs("time=1, lat=256, lon=512", metavar="<chunks>"))
        self.assertEqual(dict(chl_conc=(0, 20, 'greens'), chl_tsm=(0, 15, 'viridis')),
                         parse_cli_kwargs("chl_conc=(0,20,'greens'),chl_tsm=(0,15,'viridis')",
                                          metavar="<styles>"))

        with self.assertRaises(click.ClickException) as cm:
            parse_cli_kwargs("45 * 'A'", metavar="<chunks>")
        self.assertEqual("Invalid value for <chunks>: \"45 * 'A'\"",
                         f"{cm.exception}")

        with self.assertRaises(click.ClickException) as cm:
            parse_cli_kwargs("9==2")
        self.assertEqual("Invalid value: '9==2'",
                         f"{cm.exception}")

    def test_handle_cli_exception(self):
        self.assertEqual(1, handle_cli_exception(click.Abort(), traceback_mode=True))
        self.assertEqual(1, handle_cli_exception(click.ClickException("bad handle"), traceback_mode=True))
        self.assertEqual(2, handle_cli_exception(OSError("bad handle"), traceback_mode=True))
        self.assertEqual(3, handle_cli_exception(ValueError("bad handle"), traceback_mode=True))
        self.assertEqual(20, handle_cli_exception(click.Abort(), exit_code=20))
        self.assertEqual(20, handle_cli_exception(click.ClickException("bad handle"), exit_code=20))
        self.assertEqual(20, handle_cli_exception(OSError("bad handle"), exit_code=20))
        self.assertEqual(20, handle_cli_exception(ValueError("bad handle"), exit_code=20))
