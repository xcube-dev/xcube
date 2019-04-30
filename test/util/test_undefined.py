from unittest import TestCase

from xcube.util.undefined import UNDEFINED


class UndefinedTest(TestCase):
    def test_it(self):
        self.assertIsNotNone(UNDEFINED)
        self.assertEqual(str(UNDEFINED), 'UNDEFINED')
        self.assertEqual(repr(UNDEFINED), 'UNDEFINED')
