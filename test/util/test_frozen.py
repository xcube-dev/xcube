# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

import pytest

from xcube.util.frozen import FrozenDict, FrozenList, defrost_value, freeze_value


class FrozenDictTest(unittest.TestCase):
    FLAT = dict(a=7, b=True)
    NESTED = dict(a=7, b=True, c=FLAT, d=[FLAT, FLAT])

    def test_freeze_flat(self):
        dct = FrozenDict.freeze(self.FLAT)
        self.assertFlatOk(dct)

    def test_freeze_nested(self):
        dct = FrozenDict.freeze(self.NESTED)
        self.assertNestedOk(dct)

    def test_defrost(self):
        frozen = FrozenDict.freeze(self.NESTED)
        defrosted = frozen.defrost()

        self.assertNotIsInstance(defrosted, FrozenDict)
        self.assertIsInstance(defrosted, dict)

        self.assertNotIsInstance(defrosted["c"], FrozenDict)
        self.assertIsInstance(defrosted["c"], dict)

        self.assertNotIsInstance(defrosted["d"], FrozenList)
        self.assertIsInstance(defrosted["d"], list)

        self.assertNotIsInstance(defrosted["d"][0], FrozenDict)
        self.assertIsInstance(defrosted["d"][0], dict)

        self.assertNotIsInstance(defrosted["d"][1], FrozenDict)
        self.assertIsInstance(defrosted["d"][1], dict)

    def assertFlatOk(self, dct: dict):
        self.assertEqual(self.FLAT, dct)
        with pytest.raises(TypeError, match="dict is read-only"):
            dct["x"] = "Take this"
        with pytest.raises(TypeError, match="dict is read-only"):
            del dct["a"]
        with pytest.raises(TypeError, match="dict is read-only"):
            dct.update(x="Take this")
        with pytest.raises(TypeError, match="dict is read-only"):
            dct.pop("a")
        with pytest.raises(TypeError, match="dict is read-only"):
            dct.popitem()
        with pytest.raises(TypeError, match="dict is read-only"):
            dct.clear()

    def assertNestedOk(self, dct: dict):
        self.assertFlatOk(dct["c"])
        self.assertFlatOk(dct["d"][0])
        self.assertFlatOk(dct["d"][1])

        with pytest.raises(TypeError, match="dict is read-only"):
            dct["c"] = 3

        with pytest.raises(TypeError, match="list is read-only"):
            dct["d"][0] = 3


class FrozenListTest(unittest.TestCase):
    FLAT = ["A", "B", "C"]
    NESTED = [1, True, FLAT, [FLAT, FLAT]]

    def test_freeze_flat(self):
        frozen = FrozenList.freeze(self.FLAT)
        self.assertFlatOk(frozen)

    def test_freeze_nested(self):
        frozen = FrozenList.freeze(self.NESTED)
        self.assertNestedOk(frozen)

    def test_defrost(self):
        frozen = FrozenList.freeze(self.NESTED)
        defrosted = frozen.defrost()

        self.assertNotIsInstance(defrosted, FrozenList)
        self.assertIsInstance(defrosted, list)

        self.assertNotIsInstance(defrosted[2], FrozenList)
        self.assertIsInstance(defrosted[2], list)

        self.assertNotIsInstance(defrosted[3][0], FrozenList)
        self.assertIsInstance(defrosted[3][0], list)

        self.assertNotIsInstance(defrosted[3][1], FrozenList)
        self.assertIsInstance(defrosted[3][1], list)

    def assertFlatOk(self, lst: list):
        self.assertEqual(self.FLAT, lst)
        with pytest.raises(TypeError, match="list is read-only"):
            lst.append("X")
        with pytest.raises(TypeError, match="list is read-only"):
            lst.extend(["X"])
        with pytest.raises(TypeError, match="list is read-only"):
            lst.clear()
        with pytest.raises(TypeError, match="list is read-only"):
            lst.reverse()
        with pytest.raises(TypeError, match="list is read-only"):
            lst.sort()
        with pytest.raises(TypeError, match="list is read-only"):
            lst.remove("A")
        with pytest.raises(TypeError, match="list is read-only"):
            lst.insert(1, "X")
        with pytest.raises(TypeError, match="list is read-only"):
            lst[1] = "Take this"
        with pytest.raises(TypeError, match="list is read-only"):
            lst *= 3
        with pytest.raises(TypeError, match="list is read-only"):
            lst += ["X"]

    def assertNestedOk(self, lst: list):
        self.assertFlatOk(lst[2])
        self.assertFlatOk(lst[3][0])
        self.assertFlatOk(lst[3][1])

        with pytest.raises(TypeError, match="list is read-only"):
            lst[2] = "X"

        with pytest.raises(TypeError, match="list is read-only"):
            lst[3][0] = "X"


class FreezeValueTest(unittest.TestCase):
    def test_primitives(self):
        self.assertEqual(True, freeze_value(True))
        self.assertEqual(26, freeze_value(26))
        self.assertEqual("X", freeze_value("X"))

    def test_sequences(self):
        self.assertEqual([1, 2, 3], freeze_value([1, 2, 3]))
        self.assertIsInstance(freeze_value([1, 2, 3]), FrozenList)

        self.assertEqual([1, 2, 3], freeze_value((1, 2, 3)))
        self.assertIsInstance(freeze_value((1, 2, 3)), FrozenList)

    def test_dict(self):
        self.assertEqual({"x": 32, "y": 42}, freeze_value({"x": 32, "y": 42}))
        self.assertIsInstance(freeze_value({"x": 32, "y": 42}), FrozenDict)


class DefrostValueTest(unittest.TestCase):
    def test_primitives(self):
        self.assertEqual(True, defrost_value(True))
        self.assertEqual(26, defrost_value(26))
        self.assertEqual("X", defrost_value("X"))

    def test_frozen_dict(self):
        defrosted = defrost_value(FrozenDict({"A": 2, "B": 6}))
        self.assertNotIsInstance(defrosted, FrozenDict)
        self.assertIsInstance(defrosted, dict)
        self.assertEqual({"A": 2, "B": 6}, defrosted)

    def test_frozen_list(self):
        defrosted = defrost_value(FrozenList([1, 2, 3]))
        self.assertNotIsInstance(defrosted, FrozenList)
        self.assertIsInstance(defrosted, list)
        self.assertEqual([1, 2, 3], defrosted)
