# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.


import unittest

from zarr.storage import MemoryStore

from xcube.core.zarrstore import LoggingZarrStore


class LoggingZarrStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.zattrs_value = b''
        self.original_store = MemoryStore()
        self.original_store.update({"chl/.zattrs": self.zattrs_value})

    def test_read(self):
        logging_store = LoggingZarrStore(self.original_store)

        # noinspection PyUnresolvedReferences
        self.assertEqual([".zattrs"], logging_store.listdir("chl"))
        # noinspection PyUnresolvedReferences
        self.assertEqual(0, logging_store.getsize("chl"))
        self.assertEqual({"chl/.zattrs"}, set(logging_store.keys()))
        self.assertEqual(["chl/.zattrs"], list(iter(logging_store)))
        self.assertTrue("chl/.zattrs" in logging_store)
        self.assertEqual(1, len(logging_store))
        self.assertEqual(self.zattrs_value, logging_store.get("chl/.zattrs"))
        # assert original_store not changed
        self.assertEqual({"chl/.zattrs"}, set(self.original_store.keys()))

    def test_write(self):
        logging_store = LoggingZarrStore(self.original_store)

        zarray_value = b''
        logging_store["chl/.zarray"] = zarray_value
        self.assertEqual(
            {"chl/.zattrs", "chl/.zarray"}, set(self.original_store.keys())
        )
        del logging_store["chl/.zarray"]
        self.assertEqual({"chl/.zattrs"}, set(self.original_store.keys()))
