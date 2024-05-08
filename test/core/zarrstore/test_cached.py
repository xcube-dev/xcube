# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

import pytest
import zarr.storage

from xcube.core.zarrstore import CachedZarrStore
from xcube.core.zarrstore import DiagnosticZarrStore


class CachedZarrStoreTest(unittest.TestCase):
    def get_store(self) -> CachedZarrStore:
        self.store = {
            "chl/.zarray": b"",
            "chl/.zattrs": b"",
            "chl/0.0.0": b"",
            "chl/0.0.1": b"",
            "chl/0.1.0": b"",
            "chl/0.1.1": b"",
        }
        self.cache = DiagnosticZarrStore({})
        return CachedZarrStore(self.store, self.cache)

    def test_props(self):
        store = self.get_store()
        self.assertIsInstance(store.store, zarr.storage.BaseStore)
        self.assertIsInstance(store.cache, zarr.storage.BaseStore)

    def test_getitem(self):
        store = self.get_store()

        self.assertEqual(b"", store["chl/0.1.1"])
        self.assertEqual(
            ["__getitem__('chl/0.1.1')", "__setitem__('chl/0.1.1', bytes)"],
            self.cache.records,
        )
        self.assertIn("chl/0.1.1", self.store)
        self.assertIn("chl/0.1.1", self.cache)

        self.cache.records = []
        self.assertEqual(b"", store["chl/0.1.1"])
        self.assertEqual(["__getitem__('chl/0.1.1')"], self.cache.records)

    def test_len(self):
        store = self.get_store()
        self.assertEqual(6, len(store))

    def test_iter(self):
        store = self.get_store()
        self.assertEqual(
            [
                "chl/.zarray",
                "chl/.zattrs",
                "chl/0.0.0",
                "chl/0.0.1",
                "chl/0.1.0",
                "chl/0.1.1",
            ],
            list(iter(store)),
        )

    def test_contains(self):
        store = self.get_store()
        self.assertIn("chl/.zarray", store)
        self.assertNotIn("chl", store)

    def test_setitem(self):
        store = self.get_store()
        with pytest.raises(NotImplementedError):
            store["chl/0.0.1"] = b""

    def test_delitem(self):
        store = self.get_store()
        with pytest.raises(NotImplementedError):
            del store["chl/0.0.1"]
