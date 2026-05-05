# Copyright (c) 2018-2026 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from collections.abc import AsyncIterator, Iterable
import zarr.abc.store
import zarr.core.buffer

from xcube.util.perf import measure_time_cm


class LoggingZarrStore(zarr.abc.store.Store):
    """A Zarr Store that logs all method calls on another store *other*
    including execution time.
    """

    def __init__(self, other: zarr.abc.store.Store, logger=print, name="store"):
        super().__init__(read_only=getattr(other, "read_only", True))
        self._other = other
        self._log = logger
        self._name = name or "chunk_store"
        self._measure_time = measure_time_cm(logger=logger)

    def __eq__(self, other: object) -> bool:
        with self._measure_time(f"{self._name}.__eq__()"):
            if not isinstance(other, LoggingZarrStore):
                return False
            return self._other == other._other

    async def get(
        self,
        key: str,
        prototype: zarr.core.buffer.BufferPrototype,
        byte_range: zarr.abc.store.ByteRequest | None = None,
    ) -> zarr.core.buffer.Buffer | None:
        with self._measure_time(f"{self._name}.get({key!r}, {prototype!r}, {byte_range!r})"):
            return await self._other.get(key, prototype, byte_range=byte_range)

    async def get_partial_values(
        self,
        prototype: zarr.core.buffer.BufferPrototype,
        key_ranges: Iterable[tuple[str, tuple[int | None, int | None] | None]],
    ) -> list[zarr.core.buffer.Buffer | None]:
        with self._measure_time(f"{self._name}.get_partial_values({prototype!r}, {key_ranges!r})"):
            return await self._other.get_partial_values(prototype, key_ranges)

    async def exists(self, key: str) -> bool:
        with self._measure_time(f"{self._name}.exists({key!r})"):
            return await self._other.exists(key)

    @property
    def supports_writes(self) -> bool:
        with self._measure_time(f"{self._name}.supports_writes()"):
            return self._other.supports_writes

    async def set(self, key: str, value: zarr.core.buffer.Buffer) -> None:
        with self._measure_time(f"{self._name}.set({key!r}, {value!r})"):
            return await self._other.set(key, value)

    @property
    def supports_deletes(self) -> bool:
        with self._measure_time(f"{self._name}.supports_deletes()"):
            return self._other.supports_deletes

    async def delete(self, key: str) -> None:
        with self._measure_time(f"{self._name}.delete({key!r})"):
            return await self._other.delete(key)

    @property
    def supports_listing(self) -> bool:
        with self._measure_time(f"{self._name}.supports_listing()"):
            return self._other.supports_listing

    async def list(self) -> AsyncIterator[str]:
        with self._measure_time(f"{self._name}.list()"):
            async for k in self._other.list():
                yield k

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        with self._measure_time(f"{self._name}.list_prefix({prefix!r})"):
            async for k in self._other.list_prefix(prefix):
                yield k

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        with self._measure_time(f"{self._name}.list_dir({prefix!r})"):
            async for k in self._other.list_dir(prefix):
                yield k
