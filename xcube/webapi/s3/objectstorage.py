# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import collections.abc
from typing import Union, Tuple
from collections.abc import Iterator, Mapping

import xarray as xr
import zarr.storage

from xcube.core.mldataset import MultiLevelDataset
from xcube.server.api import ApiError


class ObjectStorage(collections.abc.Mapping):
    """Represents the emulated object storage for the S3 API.

    Keys are strings and expected to have format

    * "{dataset_id}/{level}.zarr/{*path}" for multi-level datasets and
    * "{dataset_id}/{*path}" for other datasets.

    Args:
        datasets: Mapping from dataset Identifier to (multi-level)
            datasets.
    """

    def __init__(self, datasets: Mapping[str, Union[xr.Dataset, MultiLevelDataset]]):
        self.datasets = datasets

    def __len__(self) -> int:
        return sum(1 for _ in iter(self))

    def __iter__(self) -> Iterator[str]:
        for dataset_id, dataset in self.datasets.items():
            if isinstance(dataset, MultiLevelDataset):
                for level in range(dataset.num_levels):
                    level_dataset = dataset.get_dataset(level)
                    zarr_store = level_dataset.zarr_store.get()
                    for k in zarr_store.keys():
                        yield f"{dataset_id}/{level}.zarr/{k}"

            else:
                zarr_store = dataset.zarr_store.get()
                for k in zarr_store.keys():
                    yield f"{dataset_id}/{k}"

    def __contains__(self, key: str) -> bool:
        """Overridden to avoid a call to __getitem__(),
        which will load data, but we want this to happen
        for direct __getitem__() calls only.
        """
        try:
            zarr_store, item_key = self._parse_key(key)
        except (KeyError, ApiError.NotFound):
            # Parsing failed, must be invalid key
            return False
        # Now check item_key
        return item_key in zarr_store

    def __getitem__(self, key: str) -> bytes:
        """Get bytes object for *key*."""
        zarr_store, item_key = self._parse_key(key)
        value = zarr_store[item_key]
        if not isinstance(value, bytes):
            raise RuntimeError(
                f"Zarr store of type {type(zarr_store).__name__}"
                f" must return type bytes for key {key!r},"
                f" but was {type(value).__name__}"
            )
        return value

    def _parse_key(self, key: str) -> tuple[zarr.storage.BaseStore, str]:
        """Parses a given *key* which is expected to have format
        "{dataset_id}/{level}.zarr/{*path}" for multi-level datasets and
        "{dataset_id}/{*path}" for other datasets.
        """
        try:
            dataset_id, item_key = key.split("/", maxsplit=1)
        except ValueError:
            raise KeyError(key)

        try:
            dataset = self.datasets[dataset_id]
        except ApiError as e:
            raise KeyError(key) from e

        if isinstance(dataset, MultiLevelDataset):
            try:
                level_dataset_id, item_key = item_key.split("/", maxsplit=1)
                level_dataset_id, _ = level_dataset_id.split(".", maxsplit=1)
                level = int(level_dataset_id)
            except ValueError:
                raise KeyError(key)
            if not (0 <= level < dataset.num_levels):
                raise KeyError(key)
            level_dataset = dataset.get_dataset(level)
            zarr_store = level_dataset.zarr_store.get()
        else:
            zarr_store = dataset.zarr_store.get()

        return zarr_store, item_key
