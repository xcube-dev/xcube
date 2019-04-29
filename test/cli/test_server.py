import unittest

from xcube.cli.server import main


class ServerCliSmokeTest(unittest.TestCase):

    def test_help(self):
        with self.assertRaises(SystemExit):
            main(args=["--help"])
