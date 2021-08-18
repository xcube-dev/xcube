import copy
from unittest import TestCase

from xcube.util.undefined import UNDEFINED


class UndefinedTest(TestCase):
    def test_it(self):
        self.assertIsNotNone(UNDEFINED)
        self.assertEqual(str(UNDEFINED), 'UNDEFINED')
        self.assertEqual(repr(UNDEFINED), 'UNDEFINED')
        undefined_copy = copy.deepcopy(UNDEFINED)
        self.assertEqual(UNDEFINED, undefined_copy)
        self.assertEqual(hash(UNDEFINED), hash(undefined_copy))
