# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import copy
from unittest import TestCase

from xcube.util.undefined import UNDEFINED


class UndefinedTest(TestCase):
    def test_it(self):
        self.assertIsNotNone(UNDEFINED)
        self.assertEqual(str(UNDEFINED), "UNDEFINED")
        self.assertEqual(repr(UNDEFINED), "UNDEFINED")
        undefined_copy = copy.deepcopy(UNDEFINED)
        self.assertEqual(UNDEFINED, undefined_copy)
        self.assertEqual(hash(UNDEFINED), hash(undefined_copy))
