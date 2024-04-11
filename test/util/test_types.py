# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

from xcube.util.types import normalize_scalar_or_pair


class NormalizeNumberScalarOrPairTest(unittest.TestCase):
    def test_scalar_float(self):
        self.assertEqual((1.1, 1.1), normalize_scalar_or_pair(1.1, item_type=float))
        self.assertEqual((2.0, 2.0), normalize_scalar_or_pair(2.0, item_type=float))

    def test_scalar_int(self):
        self.assertEqual((1, 1), normalize_scalar_or_pair(1, item_type=int))
        self.assertEqual((1, 1), normalize_scalar_or_pair(1, item_type=int))
        self.assertEqual((2, 2), normalize_scalar_or_pair(2, item_type=int))

    def test_pair_float(self):
        self.assertEqual(
            (1.1, 1.9), normalize_scalar_or_pair((1.1, 1.9), item_type=float)
        )
        self.assertEqual(
            (2.0, 1.9), normalize_scalar_or_pair((2.0, 1.9), item_type=float)
        )

    def test_pair_int(self):
        self.assertEqual((1, 1), normalize_scalar_or_pair((1, 1), item_type=int))
        self.assertEqual((2, 1), normalize_scalar_or_pair((2, 1), item_type=int))

    def test_wrong_type(self):
        with self.assertRaises(ValueError) as ve:
            normalize_scalar_or_pair("xyz", item_type=int, name="notastring")
        self.assertEqual(
            "notastring must be a scalar or pair of <class 'int'>, was 'xyz'",
            f"{ve.exception}",
        )

    def test_wrong_length(self):
        with self.assertRaises(ValueError) as ve:
            normalize_scalar_or_pair((0, 1, 2))
        self.assertEqual(
            "Value must be a scalar or pair of scalars, was '(0, 1, 2)'",
            f"{ve.exception}",
        )
