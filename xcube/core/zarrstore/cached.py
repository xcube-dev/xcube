# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import collections.abc
import warnings
from collections.abc import Iterator
from typing import List

import zarr.storage

from xcube.util.assertions import assert_instance, assert_true


class CachedZarrStore(zarr.storage.Store):
    """A read-only Zarr store that is faster than
    *store* because it uses a writable *cache* store.

    The *cache* store is assumed to
    read values for a given key much faster than *store*.

    Note that iterating keys and containment checks are performed
    on *store* only.

    Args:
        store: A Zarr store that is known to be slow in reading values.
        cache: A writable Zarr store that can read values faster than
            *store*.
    """

    _readable = True  # Because the base class is readable
    _listable = True  # Because the base class is listable
    _writeable = False  # Because this is not yet supported
    _erasable = False  # Because this is not yet supported

    def __init__(
        self,
        store: collections.abc.MutableMapping,
        cache: collections.abc.MutableMapping,
    ):
        assert_instance(store, collections.abc.MutableMapping, name="store")
        assert_instance(cache, collections.abc.MutableMapping, name="cache")
        if not isinstance(store, zarr.storage.BaseStore):
            store = zarr.storage.KVStore(store)
        if not isinstance(cache, zarr.storage.BaseStore):
            cache = zarr.storage.KVStore(cache)
        assert_true(store.is_readable(), message="store must be readable")
        assert_true(cache.is_readable(), message="cache must be readable")
        assert_true(cache.is_writeable(), message="cache must be writable")
        self._store = store
        self._cache = cache
        self._implement_op("listdir")
        self._implement_op("getsize")

    @property
    def store(self) -> zarr.storage.BaseStore:
        return self._store

    @property
    def cache(self) -> zarr.storage.BaseStore:
        return self._cache

    def _implement_op(self, op: str):
        if hasattr(self._store, op):
            assert hasattr(self, "_" + op)
            setattr(self, op, getattr(self, "_" + op))

    def _listdir(self, path: str = "") -> list[str]:
        # noinspection PyUnresolvedReferences
        return self._store.listdir(path=path)

    def _getsize(self, path: str) -> None:
        # noinspection PyBroadException
        try:
            # noinspection PyUnresolvedReferences
            size = self._cache.getsize(path)
        except BaseException:
            size = -1
        if size < 0:
            # noinspection PyUnresolvedReferences
            size = self._store.getsize(path)
        return size

    def __len__(self) -> int:
        return len(self._store)

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __contains__(self, key: str):
        return key in self._store

    def __getitem__(self, key: str) -> bytes:
        try:
            return self._cache[key]
        except KeyError:
            pass
        value = self._store[key]
        # noinspection PyBroadException
        try:
            self._cache[key] = value
        except BaseException as e:
            warnings.warn(f"cache write failed for key {key!r}: {e}")
        return value

    def __setitem__(self, key: str, value: bytes) -> None:
        raise NotImplementedError()

    def __delitem__(self, key: str) -> None:
        raise NotImplementedError()
