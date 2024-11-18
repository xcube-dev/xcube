# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

import pytest
import xarray as xr

from xcube.core.zarrstore.generic import GenericZarrStore
from xcube.core.zarrstore.holder import ZarrStoreHolder


class ZarrStoreHolderTest(unittest.TestCase):
    def test_zarr_store_holder_present(self):
        dataset = xr.Dataset()
        self.assertIsNotNone(dataset.zarr_store)
        self.assertIsInstance(dataset.zarr_store, ZarrStoreHolder)

    def test_zarr_store_holder_default(self):
        dataset = xr.Dataset()
        self.assertIsInstance(dataset.zarr_store.get(), GenericZarrStore)
        self.assertIs(dataset.zarr_store.get(), dataset.zarr_store.get())

    def test_zarr_store_holder_set(self):
        dataset = xr.Dataset()
        zarr_store = dict()
        dataset.zarr_store.set(zarr_store)
        self.assertIs(zarr_store, dataset.zarr_store.get())
        self.assertIs(dataset.zarr_store.get(), dataset.zarr_store.get())

    def test_zarr_store_holder_reset(self):
        dataset = xr.Dataset()
        zarr_store = dict()
        dataset.zarr_store.set(zarr_store)
        dataset.zarr_store.reset()
        self.assertIsInstance(dataset.zarr_store.get(), GenericZarrStore)
        self.assertIs(dataset.zarr_store.get(), dataset.zarr_store.get())

    # noinspection PyMethodMayBeStatic
    def test_zarr_store_type_check(self):
        dataset = xr.Dataset()
        with pytest.raises(
            TypeError,
            match="zarr_store must be an instance of"
            " <class 'collections.abc.MutableMapping'>,"
            " was <class 'int'>",
        ):
            dataset.zarr_store.set(42)
