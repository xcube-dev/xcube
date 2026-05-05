# Copyright (c) 2018-2026 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import warnings
from collections.abc import AsyncIterator, Iterable

import zarr.abc.store
import zarr.core.buffer


class CachedZarrStore(zarr.abc.store.Store):
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

    def __init__(
        self,
        store: zarr.abc.store.Store,
        cache: zarr.abc.store.Store,
    ):
        super().__init__(read_only=True)

        if not isinstance(store, zarr.abc.store.Store):
            raise TypeError("store must implement zarr v3 Store")
        if not isinstance(cache, zarr.abc.store.Store):
            raise TypeError("cache must implement zarr v3 Store")

        self._store = store
        self._cache = cache

    @property
    def store(self) -> zarr.abc.store.Store:
        return self._store

    @property
    def cache(self) -> zarr.abc.store.Store:
        return self._cache

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CachedZarrStore):
            return False
        return self._store == other._store and self._cache == other._cache

    async def get(
        self,
        key: str,
        prototype: zarr.core.buffer.BufferPrototype,
        byte_range: zarr.abc.store.ByteRequest | None = None,
    ) -> zarr.core.buffer.Buffer | None:
        # Try cache first
        if await self._cache.exists(key):
            return await self._cache.get(key, prototype, byte_range=byte_range)

        # Fallback to slow store
        value = await self._store.get(key, prototype, byte_range=byte_range)

        # Populate cache (only for full reads)
        if value is not None and byte_range is None:
            try:
                await self._cache.set(key, value)
            except Exception as e:
                warnings.warn(f"cache write failed for key {key!r}: {e}")

        return value

    async def get_partial_values(
        self,
        prototype: zarr.core.buffer.BufferPrototype,
        key_ranges: Iterable[tuple[str, tuple[int | None, int | None] | None]],
    ) -> list[zarr.core.buffer.Buffer | None]:
        results: list[zarr.core.buffer.Buffer | None] = []
        for key, byte_range in key_ranges:
            value = await self.get(key, prototype, byte_range=byte_range)
            results.append(value)
        return results

    async def exists(self, key: str) -> bool:
        return await self._store.exists(key)

    @property
    def supports_writes(self) -> bool:
        return False

    async def set(self, key: str, value: zarr.core.buffer.Buffer) -> None:
        raise NotImplementedError("Read-only store")

    @property
    def supports_deletes(self) -> bool:
        return False

    async def delete(self, key: str) -> None:
        raise NotImplementedError("Read-only store")

    @property
    def supports_listing(self) -> bool:
        return self._store.supports_listing

    async def list(self) -> AsyncIterator[str]:
        if not self._store.supports_listing:
            raise NotImplementedError("Underlying store does not support listing")
        async for key in self._store.list():
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        if not self._store.supports_listing:
            raise NotImplementedError("Underlying store does not support list_prefix")

        async for key in self._store.list_prefix(prefix):
            yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        if not self._store.supports_listing:
            raise NotImplementedError("Underlying store does not support listing")

        async for key in self._store.list_dir(prefix):
            yield key
