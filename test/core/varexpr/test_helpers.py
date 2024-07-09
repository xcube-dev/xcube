# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest


from xcube.core.varexpr import split_var_assignment


class HelpersTest(unittest.TestCase):
    def test_split_var_assignment(self):
        self.assertEqual(("A", None), split_var_assignment("A"))
        self.assertEqual(("A", "B"), split_var_assignment("A=B"))
        self.assertEqual(("A", "B + C"), split_var_assignment(" A = B + C "))
