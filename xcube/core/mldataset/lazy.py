# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import threading
import uuid
from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from typing import Any, Dict, Optional

import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.util.assertions import assert_instance

from .abc import MultiLevelDataset


class LazyMultiLevelDataset(MultiLevelDataset, metaclass=ABCMeta):
    """A multi-level dataset where each level dataset is lazily retrieved,
    i.e. read or computed by the abstract method
    ``get_dataset_lazily(index, **kwargs)``.

    Args:
        ds_id: Optional dataset identifier.
        parameters: Optional keyword arguments that will be passed to
            the ``get_dataset_lazily`` method.
    """

    def __init__(
        self,
        grid_mapping: Optional[GridMapping] = None,
        num_levels: Optional[int] = None,
        ds_id: Optional[str] = None,
        parameters: Optional[Mapping[str, Any]] = None,
    ):
        if grid_mapping is not None:
            assert_instance(grid_mapping, GridMapping, name="grid_mapping")
        if ds_id is not None:
            assert_instance(ds_id, str, name="ds_id")
        self._grid_mapping = grid_mapping
        self._num_levels = num_levels
        self._ds_id = ds_id
        self._level_datasets: dict[int, xr.Dataset] = {}
        self._parameters = parameters or {}
        self._lock = threading.RLock()

    @property
    def ds_id(self) -> str:
        if self._ds_id is None:
            with self._lock:
                self._ds_id = str(uuid.uuid4())
        return self._ds_id

    @ds_id.setter
    def ds_id(self, ds_id: str):
        assert_instance(ds_id, str, name="ds_id")
        self._ds_id = ds_id

    @property
    def grid_mapping(self) -> GridMapping:
        if self._grid_mapping is None:
            with self._lock:
                self._grid_mapping = self._get_grid_mapping_lazily()
        return self._grid_mapping

    @property
    def num_levels(self) -> int:
        if self._num_levels is None:
            with self._lock:
                self._num_levels = self._get_num_levels_lazily()
        return self._num_levels

    @property
    def lock(self) -> threading.RLock:
        """Get the reentrant lock used by this object to synchronize
        lazy instantiation of properties.
        """
        return self._lock

    def get_dataset(self, index: int) -> xr.Dataset:
        """Get or compute the dataset for the level at given *index*.

        Args:
            index: the level index

        Returns:
            the dataset for the level at *index*.
        """
        if index not in self._level_datasets:
            with self._lock:
                # noinspection PyTypeChecker
                level_dataset = self._get_dataset_lazily(index, self._parameters)
                self.set_dataset(index, level_dataset)
        # noinspection PyTypeChecker
        return self._level_datasets[index]

    def set_dataset(self, index: int, level_dataset: xr.Dataset):
        """Set the dataset for the level at given *index*.

        Callers need to ensure that the given *level_dataset*
        has the correct spatial dimension sizes for the
        given level at *index*.

        Args:
            index: the level index
            level_dataset: the dataset for the level at *index*.
        """
        with self._lock:
            self._level_datasets[index] = level_dataset

    @abstractmethod
    def _get_num_levels_lazily(self) -> int:
        """Retrieve, i.e. read or compute, the number of levels.

        Returns:
            the number of dataset levels.
        """

    @abstractmethod
    def _get_dataset_lazily(self, index: int, parameters: dict[str, Any]) -> xr.Dataset:
        """Retrieve, i.e. read or compute, the dataset for the
        level at given *index*.

        Args:
            index: the level index
            parameters: *parameters* keyword argument that was passed to
                constructor.

        Returns:
            the dataset for the level at *index*.
        """

    def _get_grid_mapping_lazily(self) -> GridMapping:
        """Retrieve, i.e. read or compute, the tile grid used
        by the multi-level dataset.

        Returns:
            the dataset for the level at *index*.
        """
        return GridMapping.from_dataset(self.get_dataset(0))

    def close(self):
        with self._lock:
            for dataset in self._level_datasets.values():
                if dataset is not None:
                    dataset.close()
