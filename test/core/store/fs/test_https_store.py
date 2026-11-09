# Copyright (c) 2018-2026 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest
from unittest.mock import patch

from fsspec.implementations.http import HTTPFileSystem

from xcube.core.store import new_data_store

# A fake web server: directory URLs map to their HTML-parsed listing,
# just like fsspec's HTTPFileSystem returns them (full URLs,
# directories with a trailing slash).
LISTINGS = {
    "https://test.org/data": [
        {"name": "https://test.org/data/cube.zarr/", "type": "directory"},
        {"name": "https://test.org/data/sst.nc", "type": "file"},
        {"name": "https://test.org/data/README.txt", "type": "file"},
        {"name": "https://test.org/data/sub/", "type": "directory"},
    ],
    "https://test.org/data/sub": [
        {"name": "https://test.org/data/sub/chl.nc", "type": "file"},
    ],
}


def _exists(self, path, **kwargs):
    # HTTPFileSystem cannot resolve paths without a URL scheme.
    return path.rstrip("/") in LISTINGS or any(
        info["name"].rstrip("/") == path.rstrip("/")
        for listing in LISTINGS.values()
        for info in listing
    )


def _ls(self, path, detail=True, **kwargs):
    if path.rstrip("/") not in LISTINGS:
        raise FileNotFoundError(path)
    return [dict(info, size=None) for info in LISTINGS[path.rstrip("/")]]


@patch.object(HTTPFileSystem, "exists", _exists)
@patch.object(HTTPFileSystem, "ls", _ls)
class HttpsFsDataStoreTest(unittest.TestCase):
    def test_list_data_ids(self):
        store = new_data_store("https", root="test.org/data")
        self.assertEqual("test.org/data", store.root)
        self.assertEqual(["cube.zarr", "sst.nc"], sorted(store.list_data_ids()))

    def test_list_data_ids_with_depth(self):
        store = new_data_store("https", root="test.org/data", max_depth=2)
        self.assertEqual(
            ["cube.zarr", "sst.nc", "sub/chl.nc"], sorted(store.list_data_ids())
        )

    def test_root_given_as_url(self):
        store = new_data_store("https", root="https://test.org/data/")
        self.assertEqual("test.org/data", store.root)
        self.assertEqual(["cube.zarr", "sst.nc"], sorted(store.list_data_ids()))
        self.assertTrue(store.has_data("sst.nc"))
