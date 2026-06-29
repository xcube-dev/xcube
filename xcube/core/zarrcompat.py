# Copyright (c) 2018-2026 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import collections.abc
import os.path
from logging import Logger
from typing import Any, Iterator, Optional

import fsspec
import xarray as xr
import zarr

from xcube.constants import LOG, LOG_LEVEL_TRACE

DEFAULT_ZARR_FORMAT = 2

_ZARR_FORMAT_PARAM_NAMES = ("zarr_format", "zarr_version")


class BytesMutableMapping(collections.abc.MutableMapping):
    """Mutable bytes mapping suitable for materialized Zarr v2 stores."""

    def __init__(self, data: Optional[dict[str, bytes]] = None):
        self._data = data if data is not None else {}

    def __getitem__(self, key: str) -> bytes:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = _to_bytes(value)

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


class GenericZarrStore(BytesMutableMapping):
    """Materialized in-memory Zarr v2 store for dataset fallbacks."""

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> "GenericZarrStore":
        mapping = dataset_to_zarr_mapping(dataset)
        return cls(dict(mapping.items()))


class LoggingZarrStore(collections.abc.MutableMapping):
    """A small logging wrapper for mapping-style Zarr stores."""

    def __init__(
        self,
        other: collections.abc.MutableMapping,
        logger: Logger = LOG,
        name: Optional[str] = None,
    ):
        self._other = other
        self._logger = logger
        self._name = name or f"{type(other).__name__}"

    def keys(self):
        self._trace("keys()")
        return self._other.keys()

    def listdir(self, key: str):
        self._trace(f"listdir({key!r})")
        if hasattr(self._other, "listdir"):
            return self._other.listdir(key)
        prefix = f"{key}/" if key else ""
        prefix_len = len(prefix)
        return sorted(
            k[prefix_len:].split("/", maxsplit=1)[0]
            for k in self._other.keys()
            if k.startswith(prefix)
        )

    def getsize(self, key: str) -> int:
        self._trace(f"getsize({key!r})")
        if hasattr(self._other, "getsize"):
            return self._other.getsize(key)
        return len(self._other[key])

    def __getitem__(self, key: str) -> bytes:
        self._trace(f"__getitem__({key!r})")
        return self._other[key]

    def __setitem__(self, key: str, value: bytes) -> None:
        self._trace(f"__setitem__({key!r}, ...)")
        self._other[key] = value

    def __delitem__(self, key: str) -> None:
        self._trace(f"__delitem__({key!r})")
        del self._other[key]

    def __iter__(self) -> Iterator[str]:
        self._trace("__iter__()")
        return iter(self._other)

    def __len__(self) -> int:
        self._trace("__len__()")
        return len(self._other)

    def _trace(self, message: str) -> None:
        self._logger.log(LOG_LEVEL_TRACE, f"{self._name}.{message}")


def get_zarr_format(params: dict[str, Any], default: int | None = None) -> int | None:
    for name in _ZARR_FORMAT_PARAM_NAMES:
        if name in params and params[name] is not None:
            return params[name]
    return default


def normalize_zarr_format_params(
    params: dict[str, Any], default: int | None = None
) -> dict[str, Any]:
    params = dict(params)
    zarr_format = get_zarr_format(params, default=default)
    if "zarr_version" in params and "zarr_format" not in params:
        params["zarr_format"] = params.pop("zarr_version")
    if zarr_format is not None:
        params.setdefault("zarr_format", zarr_format)
    return params


def pop_zarr_write_options(
    params: dict[str, Any],
    default_format: int = DEFAULT_ZARR_FORMAT,
) -> tuple[dict[str, Any], int, bool]:
    params = normalize_zarr_format_params(params, default=default_format)
    zarr_format = int(params["zarr_format"])
    if zarr_format not in (2, 3):
        raise ValueError("zarr_format must be 2 or 3")
    consolidated_given = "consolidated" in params
    consolidated = params.pop("consolidated", None)
    if consolidated is None:
        consolidated = zarr_format == 2
    elif consolidated and zarr_format == 3 and consolidated_given:
        raise ValueError("consolidated=True is not supported for Zarr format 3")
    return params, zarr_format, bool(consolidated)


def new_zarr_store(
    fs: fsspec.AbstractFileSystem,
    path: str,
    *,
    mode: str = "r",
    log_access: bool = False,
    cache_size: int | None = None,
    name: str | None = None,
) -> tuple[Any, collections.abc.MutableMapping]:
    """Return `(xarray_store, mapping_store)` for a filesystem path."""

    create = mode != "r"
    mapper = fs.get_mapper(path, create=create)
    mapping: collections.abc.MutableMapping = mapper
    if log_access:
        mapping = LoggingZarrStore(mapping, name=name)

    if hasattr(zarr.storage, "FsspecStore"):
        xarray_store = zarr.storage.FsspecStore.from_mapper(
            mapper, read_only=mode == "r"
        )
    else:
        xarray_store = mapping
        if (
            isinstance(cache_size, int)
            and cache_size > 0
            and hasattr(zarr, "LRUStoreCache")
        ):
            xarray_store = zarr.LRUStoreCache(xarray_store, max_size=cache_size)

    return xarray_store, mapping


def has_consolidated_metadata(fs: fsspec.AbstractFileSystem, path: str) -> bool:
    return fs.exists(f"{path}/.zmetadata")


def detect_zarr_format(path_or_store: Any) -> int | None:
    if isinstance(path_or_store, str):
        if os.path.exists(os.path.join(path_or_store, ".zgroup")):
            return 2
        if os.path.exists(os.path.join(path_or_store, "zarr.json")):
            return 3
    elif isinstance(path_or_store, collections.abc.Mapping):
        if ".zgroup" in path_or_store:
            return 2
        if "zarr.json" in path_or_store:
            return 3
    return None


def is_zarr_path(path: str) -> bool:
    return (
        os.path.isfile(os.path.join(path, ".zgroup"))
        or os.path.isfile(os.path.join(path, "zarr.json"))
    )


def consolidate_zarr_metadata(path_or_store: Any, zarr_format: int | None = None):
    zarr_format = zarr_format or detect_zarr_format(path_or_store)
    if zarr_format == 3:
        return None
    return zarr.consolidate_metadata(path_or_store)


def open_zarr_group(path_or_store: Any, mode: str = "a", zarr_format: int | None = None):
    kwargs = {}
    if zarr_format is not None:
        kwargs["zarr_format"] = zarr_format
    return zarr.open_group(path_or_store, mode=mode, **kwargs)


def open_zarr_array(path_or_store: Any, mode: str = "a"):
    return zarr.open_array(path_or_store, mode=mode)


def save_zarr_array(path_or_store: Any, data: Any, **kwargs):
    return zarr.save_array(path_or_store, data, **kwargs)


def dataset_to_zarr_mapping(dataset: xr.Dataset) -> collections.abc.MutableMapping:
    store = BytesMutableMapping()
    dataset.to_zarr(
        store,
        mode="w",
        consolidated=True,
        zarr_format=DEFAULT_ZARR_FORMAT,
    )
    return store


def make_blosc_codec(compressor: dict[str, Any], zarr_format: int):
    compressor = dict(compressor)
    if zarr_format == 3:
        shuffle = compressor.get("shuffle")
        if isinstance(shuffle, int):
            compressor["shuffle"] = {
                0: "noshuffle",
                1: "shuffle",
                2: "bitshuffle",
            }.get(shuffle, "shuffle")
        return zarr.codecs.BloscCodec(**compressor)

    import numcodecs

    return numcodecs.Blosc(**compressor)


def normalize_bytes_mapping_value(value: Any) -> bytes:
    return _to_bytes(value)


def _to_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if hasattr(value, "to_bytes"):
        return value.to_bytes()
    if isinstance(value, str):
        return value.encode("utf-8")
    return bytes(value)
