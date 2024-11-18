# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.


import collections.abc
from logging import Logger
from typing import Optional
from collections.abc import Iterator, Iterable, KeysView

import zarr.storage

from xcube.constants import LOG
from xcube.util.assertions import assert_instance
from xcube.util.perf import measure_time_cm


class LoggingZarrStore(zarr.storage.BaseStore):
    """A Zarr Store that logs all method calls on another store *other*
    including execution time.
    """

    def __init__(
        self,
        other: collections.abc.MutableMapping,
        logger: Logger = LOG,
        name: Optional[str] = None,
    ):
        assert_instance(other, collections.abc.MutableMapping)
        self._other = other
        self._measure_time = measure_time_cm(logger=logger)
        self._name = name or "chunk_store"
        if hasattr(other, "listdir"):
            setattr(self, "listdir", self.__listdir)
        if hasattr(other, "getsize"):
            setattr(self, "getsize", self.__getsize)

    def __listdir(self, key: str) -> Iterable[str]:
        with self._measure_time(f"{self._name}.listdir({key!r})"):
            # noinspection PyUnresolvedReferences
            return self._other.listdir(key)

    def __getsize(self, key: str) -> int:
        with self._measure_time(f"{self._name}.getsize({key!r})"):
            # noinspection PyUnresolvedReferences
            return self._other.getsize(key)

    def keys(self) -> KeysView[str]:
        with self._measure_time(f"{self._name}.keys()"):
            # noinspection PyTypeChecker
            return self._other.keys()

    def __iter__(self) -> Iterator[str]:
        with self._measure_time(f"{self._name}.__iter__()"):
            return self._other.__iter__()

    def __len__(self) -> int:
        with self._measure_time(f"{self._name}.__len__()"):
            return self._other.__len__()

    def __contains__(self, key) -> bool:
        with self._measure_time(f"{self._name}.__contains__({key!r})"):
            return self._other.__contains__(key)

    def __getitem__(self, key: str) -> bytes:
        with self._measure_time(f"{self._name}.__getitem__({key!r})"):
            return self._other.__getitem__(key)

    def __setitem__(self, key: str, value: bytes) -> None:
        with self._measure_time(f"{self._name}.__setitem__({key!r}, <value>)"):
            return self._other.__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        with self._measure_time(f"{self._name}.__delitem__({key!r})"):
            return self._other.__delitem__(key)
