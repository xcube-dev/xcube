# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

from test.mixins import AlmostEqualDeepMixin


class AlmostEqualDeepMixinTest(unittest.TestCase, AlmostEqualDeepMixin):
    def test_int_and_float_7_places_default(self):
        self.assertAlmostEqualDeep(0, 0.8e-8)
        with self.assertRaises(AssertionError):
            self.assertAlmostEqualDeep(0, 0.8e-7)

    def test_int(self):
        self.assertAlmostEqualDeep(45, 45)
        with self.assertRaises(AssertionError):
            self.assertAlmostEqualDeep(45, 54)

    def test_str(self):
        self.assertAlmostEqualDeep("abc", "abc")
        with self.assertRaises(AssertionError):
            self.assertAlmostEqualDeep("abc", "Abc")

    def test_bool(self):
        self.assertAlmostEqualDeep(True, True)
        with self.assertRaises(AssertionError):
            self.assertAlmostEqualDeep(True, False)

    def test_set(self):
        expected = {"a", 1.1256, True}
        self.assertAlmostEqualDeep(expected, expected)
        self.assertAlmostEqualDeep(expected, {"a", 1.1256, True})
        with self.assertRaises(AssertionError):
            # We currently don't test sets
            self.assertAlmostEqualDeep(expected, {"a", 1.1251, True}, places=2)
        with self.assertRaises(AssertionError):
            self.assertAlmostEqualDeep(expected, {"a", 1.1256, False})

    def test_dict(self):
        expected = {"a": 1.1256, "b": 5}
        self.assertAlmostEqualDeep(expected, expected)
        self.assertAlmostEqualDeep(expected, {"a": 1.1256, "b": 5})
        self.assertAlmostEqualDeep(expected, {"a": 1.1251, "b": 5}, places=3)
        with self.assertRaises(AssertionError):
            self.assertAlmostEqualDeep(expected, {"a": 1.1251, "b": 5}, places=4)
        with self.assertRaises(AssertionError):
            self.assertAlmostEqualDeep(expected, {"a": 1.1256, "b": 6})
        with self.assertRaises(AssertionError):
            self.assertAlmostEqualDeep(expected, [1, 2, 3])
        with self.assertRaises(AssertionError):
            self.assertAlmostEqualDeep(expected, 3456)

    def test_list(self):
        expected = ["a", 1.1256, True]
        self.assertAlmostEqualDeep(expected, expected)
        self.assertAlmostEqualDeep(expected, ["a", 1.1256, True])
        self.assertAlmostEqualDeep(expected, ("a", 1.1256, True))
        self.assertAlmostEqualDeep(expected, ["a", 1.1251, True], places=3)
        with self.assertRaises(AssertionError):
            self.assertAlmostEqualDeep(expected, ["a", 1.1251, True], places=4)
        with self.assertRaises(AssertionError):
            self.assertAlmostEqualDeep(expected, ["a", 1.1256, False], places=4)
        with self.assertRaises(AssertionError):
            self.assertAlmostEqualDeep(expected, [1, 2, 3])
        with self.assertRaises(AssertionError):
            self.assertAlmostEqualDeep(expected, 3456)

    def test_list_dict_tuple(self):
        expected = [
            {"a": True, "b": (1.1256, 45, True)},
            {"a": False, "b": (2.1256, 46, False)},
        ]
        self.assertAlmostEqualDeep(expected, expected)
        self.assertAlmostEqualDeep(
            expected,
            [
                {"a": True, "b": (1.1256, 45, True)},
                {"a": False, "b": (2.1256, 46, False)},
            ],
        )
        self.assertAlmostEqualDeep(
            expected,
            [
                {"a": True, "b": (1.1251, 45, True)},
                {"a": False, "b": (2.1259, 46, False)},
            ],
            places=3,
        )
        with self.assertRaises(AssertionError):
            self.assertAlmostEqualDeep(
                expected,
                [
                    {"a": True, "b": (1.1251, 45, True)},
                    {"a": False, "b": (2.1259, 46, False)},
                ],
                places=4,
            )
