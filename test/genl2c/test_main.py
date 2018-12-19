import unittest

from xcube.genl2c.main import main


class MainTest(unittest.TestCase):
    def test_main_succeeds(self):
        with self.assertRaises(SystemExit) as cm:
            main(['--help'])
        self.assertEqual("0", f"{cm.exception}")

    def test_main_fails(self):
        with self.assertRaises(SystemExit) as cm:
            main(['--helpi'])
        self.assertEqual("2", f"{cm.exception}")



