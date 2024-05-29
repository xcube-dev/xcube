# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import collections.abc
from typing import List
from collections.abc import Iterator

import zarr.storage


class DiagnosticZarrStore(zarr.storage.Store):
    """A diagnostic Zarr store used for testing and investigating
    behaviour of Zarr and xarray's Zarr backend.

    Args:
        store: Wrapped Zarr store.
    """

    def __init__(self, store: collections.abc.MutableMapping):
        self.store = store
        self.records: list[str] = []
        if hasattr(self.store, "listdir"):
            self.listdir = self._listdir
        if hasattr(self.store, "getsize"):
            self.getsize = self._getsize

    def _add_record(self, record: str):
        self.records.append(record)

    def _listdir(self, path: str = "") -> list[str]:
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
