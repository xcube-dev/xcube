# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

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
        self.assertIs(dataset.zarr_store.get(),
                      dataset.zarr_store.get())

    def test_zarr_store_holder_set(self):
        dataset = xr.Dataset()
        zarr_store = dict()
        dataset.zarr_store.set(zarr_store)
        self.assertIs(zarr_store, dataset.zarr_store.get())
        self.assertIs(dataset.zarr_store.get(),
                      dataset.zarr_store.get())

    def test_zarr_store_holder_reset(self):
        dataset = xr.Dataset()
        zarr_store = dict()
        dataset.zarr_store.set(zarr_store)
        dataset.zarr_store.reset()
        self.assertIsInstance(dataset.zarr_store.get(), GenericZarrStore)
        self.assertIs(dataset.zarr_store.get(),
                      dataset.zarr_store.get())

    # noinspection PyMethodMayBeStatic
    def test_zarr_store_type_check(self):
        dataset = xr.Dataset()
        with pytest.raises(TypeError,
                           match="zarr_store must be an instance of"
                                 " <class 'collections.abc.MutableMapping'>,"
                                 " was <class 'int'>"):
            dataset.zarr_store.set(42)
