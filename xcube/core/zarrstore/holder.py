# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import collections.abc
import threading
import warnings
from typing import Optional

import xarray as xr

from xcube.constants import LOG
from xcube.util.assertions import assert_instance


@xr.register_dataset_accessor("zarr_store")
class ZarrStoreHolder:
    """Represents a xarray dataset property ``zarr_store``.

    It is used to permanently associate a dataset with its
    Zarr store, which would otherwise not be possible.

    In xcube server, we use the new property to expose
    datasets via the S3 emulation API.

    For that concept to work, datasets must be associated
    with their Zarr stores explicitly.
    Therefore, the xcube data store framework sets the
    Zarr stores of datasets after opening them ``xr.open_zarr()``:::

        dataset = xr.open_zarr(zarr_store, **open_params)
        dataset.zarr_store.set(zarr_store)

    Note, that the dataset may change after the Zarr store has been set,
    so that the dataset and its Zarr store are no longer in sync.
    This may be an issue and limit the application of the new property.

    Args:
        dataset: The xarray dataset that is associated with a Zarr
            store.
    """

    def __init__(self, dataset: xr.Dataset):
        self._dataset = dataset
        self._zarr_store: Optional[collections.abc.MutableMapping] = None
        self._lock = threading.RLock()

    def get(self) -> collections.abc.MutableMapping:
        """Get the Zarr store of a dataset.
        If no Zarr store has been set, the method will use
        ``GenericZarrStore.from_dataset()`` to create and set
        one.

        Returns:
            The Zarr store.
        """
        if self._zarr_store is None:
            # Double-checked locking pattern
            with self._lock:
                if self._zarr_store is None:
                    from xcube.core.zarrstore import GenericZarrStore

                    self._zarr_store = GenericZarrStore.from_dataset(self._dataset)
                    source = self._dataset.encoding.get("source", "?")
                    LOG.warning(
                        f"dataset {source!r} is assigned a"
                        f" GenericZarrStore which may introduce"
                        f" performance penalties"
                    )
        return self._zarr_store

    def set(self, zarr_store: collections.abc.MutableMapping) -> None:
        """Set the Zarr store of a dataset.

        Args:
            zarr_store: The Zarr store.
        """
        assert_instance(zarr_store, collections.abc.MutableMapping, name="zarr_store")
        with self._lock:
            self._zarr_store = zarr_store

    def reset(self) -> None:
        """Resets the Zarr store."""
        with self._lock:
            self._zarr_store = None
