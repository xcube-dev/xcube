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

import collections.abc
from typing import Iterator, List

import zarr.storage


class DiagnosticZarrStore(zarr.storage.Store):
    """A diagnostic Zarr store used for testing and investigating
    behaviour of Zarr and xarray's Zarr backend.

    :param store: Wrapped Zarr store.
    """

    def __init__(self, store: collections.abc.MutableMapping):
        self.store = store
        self.records: List[str] = []
        if hasattr(self.store, "listdir"):
            self.listdir = self._listdir
        if hasattr(self.store, "getsize"):
            self.getsize = self._getsize

    def _add_record(self, record: str):
        self.records.append(record)

    def _listdir(self, path: str = "") -> List[str]:
        self._add_record(f"listdir({path!r})")
        # noinspection PyUnresolvedReferences
        return self.store.listdir(path=path)

    def _getsize(self, key: str) -> int:
        self._add_record(f"getsize({key!r})")
        # noinspection PyUnresolvedReferences
        return self.store.getsize(key)

    def keys(self):
        self._add_record("keys()")
        return self.store.keys()

    def __setitem__(self, key: str, value: bytes) -> None:
        self._add_record(f"__setitem__({key!r}, {type(value).__name__})")
        self.store.__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        self._add_record(f"__delitem__({key!r})")
        self.store.__delitem__(key)

    def __getitem__(self, key: str) -> bytes:
        self._add_record(f"__getitem__({key!r})")
        return self.store.__getitem__(key)

    def __len__(self) -> int:
        self._add_record("__len__()")
        return self.store.__len__()

    def __iter__(self) -> Iterator[str]:
        self._add_record("__iter__()")
        return self.store.__iter__()
