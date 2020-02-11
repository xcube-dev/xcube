import unittest

import click

from xcube.cli.common import assert_positive_int_item
from xcube.cli.common import handle_cli_exception
from xcube.cli.common import parse_cli_kwargs
from xcube.cli.common import parse_cli_sequence


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

    def test_parse_cli_sequence(self):
        self.assertEqual((512, 1024),
                         parse_cli_sequence('512, 1024',
                                            metavar='X',
                                            item_parser=int))
        self.assertEqual((11, 11, 11),
                         parse_cli_sequence('11',
                                            metavar='X',
                                            item_parser=int,
                                            num_items=3))
        self.assertEqual((11, 12, 13),
                         parse_cli_sequence(['11', '12 ', ' 13'],
                                            metavar='X',
                                            item_parser=int))
        self.assertEqual(None,
                         parse_cli_sequence(None, allow_none=True))
        with self.assertRaises(click.ClickException) as cm:
            parse_cli_sequence(None,
                               metavar='X',
                               allow_none=False)
        self.assertEqual("X must be given", f'{cm.exception}')
        with self.assertRaises(click.ClickException) as cm:
            parse_cli_sequence('11',
                               metavar='X',
                               item_parser=int,
                               num_items_min=3,
                               item_plural_name='lollies')
        self.assertEqual("X must have at least 3 lollies separated by ','", f'{cm.exception}')
        with self.assertRaises(click.ClickException) as cm:
            parse_cli_sequence('11,12,13',
                               metavar='X',
                               item_parser=int,
                               num_items_max=2,
                               item_plural_name='lollies')
        self.assertEqual("X must have no more than 2 lollies separated by ','", f'{cm.exception}')
        with self.assertRaises(click.ClickException) as cm:
            parse_cli_sequence('11,12,13',
                               metavar='X',
                               item_parser=int,
                               num_items=2,
                               item_plural_name='lollies')
        self.assertEqual("X must have 2 lollies separated by ','", f'{cm.exception}')
        with self.assertRaises(click.ClickException) as cm:
            parse_cli_sequence('11,,13',
                               metavar='X',
                               item_parser=int,
                               num_items=3,
                               item_plural_name='lollies',
                               allow_empty_items=False)
        self.assertEqual("lollies in X must not be empty", f'{cm.exception}')
        with self.assertRaises(click.ClickException) as cm:
            parse_cli_sequence('11,12,-13',
                               metavar='X',
                               item_parser=int,
                               item_validator=assert_positive_int_item,
                               num_items=3,
                               item_plural_name='lollies')
        self.assertEqual("Invalid lollies in X found: all items must be positive integer numbers", f'{cm.exception}')
        with self.assertRaises(ValueError) as cm:
            parse_cli_sequence('11,12,-13',
                               metavar='X',
                               item_parser=int,
                               item_validator=assert_positive_int_item,
                               num_items=3,
                               item_plural_name='lollies',
                               error_type=ValueError)
        self.assertEqual("Invalid lollies in X found: all items must be positive integer numbers", f'{cm.exception}')

    def test_handle_cli_exception(self):
        self.assertEqual(1, handle_cli_exception(click.Abort(), traceback_mode=True))
        self.assertEqual(1, handle_cli_exception(click.ClickException("bad handle"), traceback_mode=True))
        self.assertEqual(2, handle_cli_exception(OSError("bad handle"), traceback_mode=True))
        self.assertEqual(3, handle_cli_exception(ValueError("bad handle"), traceback_mode=True))
        self.assertEqual(20, handle_cli_exception(click.Abort(), exit_code=20))
        self.assertEqual(20, handle_cli_exception(click.ClickException("bad handle"), exit_code=20))
        self.assertEqual(20, handle_cli_exception(OSError("bad handle"), exit_code=20))
        self.assertEqual(20, handle_cli_exception(ValueError("bad handle"), exit_code=20))
