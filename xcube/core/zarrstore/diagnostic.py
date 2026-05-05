# Copyright (c) 2018-2026 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from collections.abc import AsyncIterator, Iterable

import zarr.abc.store
import zarr.core.buffer


class DiagnosticZarrStore(zarr.abc.store.Store):
    """A diagnostic Zarr store used for testing and investigating
    behaviour of Zarr and xarray's Zarr backend.

    Args:
        store: Wrapped Zarr store.
    """

    def __init__(
        self,
        store: zarr.abc.store.Store,
    ):
        super().__init__(read_only=True)
        self._store = store
        self.records: list[str] = []

    @property
    def store(self) -> zarr.abc.store.Store:
        return self._store

    def _add_record(self, record: str):
        self.records.append(record)

    def __eq__(self, other: object) -> bool:
        self._add_record(f"__eq__({other!r})")
        if not isinstance(other, DiagnosticZarrStore):
            return False
        return self._store == other._store

    async def get(
        self,
        key: str,
        prototype: zarr.core.buffer.BufferPrototype,
        byte_range: zarr.abc.store.ByteRequest | None = None,
    ) -> zarr.core.buffer.Buffer | None:
        self._add_record(f"get({key!r}, {prototype!r}, {byte_range!r})")
        value = await self._store.get(key, prototype, byte_range=byte_range)
        return value

    async def get_partial_values(
        self,
        prototype: zarr.core.buffer.BufferPrototype,
        key_ranges: Iterable[tuple[str, tuple[int | None, int | None] | None]],
    ) -> list[zarr.core.buffer.Buffer | None]:
        self._add_record(f"get_partial_values({prototype!r}, {key_ranges!r})")
        results: list[zarr.core.buffer.Buffer | None] = []
        for key, byte_range in key_ranges:
            value = await self.get(key, prototype, byte_range=byte_range)
            results.append(value)
        return results

    async def exists(self, key: str) -> bool:
        self._add_record(f"exists({key!r})")
        return await self._store.exists(key)

    @property
    def supports_writes(self) -> bool:
        self._add_record("supports_writes()")
        return False

    async def set(self, key: str, value: zarr.core.buffer.Buffer) -> None:
        raise NotImplementedError("Read-only store")

    @property
    def supports_deletes(self) -> bool:
        self._add_record("supports_deletes()")
        return False

    async def delete(self, key: str) -> None:
        raise NotImplementedError("Read-only store")

    @property
    def supports_listing(self) -> bool:
        self._add_record("supports_listing()")
        return self._store.supports_listing

    async def list(self) -> AsyncIterator[str]:
        self._add_record("list()")
        if not self._store.supports_listing:
            raise NotImplementedError("Underlying store does not support listing")
        async for key in self._store.list():
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        self._add_record(f"list_prefix({prefix!r})")
        if not self._store.supports_listing:
            raise NotImplementedError("Underlying store does not support list_prefix")

        async for key in self._store.list_prefix(prefix):
            yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        self._add_record(f"list_dir({prefix!r})")
        if not self._store.supports_listing:
            raise NotImplementedError("Underlying store does not support listing")

        async for key in self._store.list_dir(prefix):
            yield key
