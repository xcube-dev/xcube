# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Any, Callable, Dict

import xarray as xr

from .abc import MultiLevelDataset
from .lazy import LazyMultiLevelDataset


class MappedMultiLevelDataset(LazyMultiLevelDataset):
    def __init__(
        self,
        ml_dataset: MultiLevelDataset,
        mapper_function: Callable[[xr.Dataset], xr.Dataset],
        ds_id: str = None,
        mapper_params: dict[str, Any] = None,
    ):
        """"""
        super().__init__(ds_id=ds_id, parameters=mapper_params)
        self._ml_dataset = ml_dataset
        self._mapper_function = mapper_function

    def _get_num_levels_lazily(self) -> int:
        return self._ml_dataset.num_levels

    def _get_dataset_lazily(
        self, index: int, mapper_params: dict[str, Any]
    ) -> xr.Dataset:
        return self._mapper_function(
            self._ml_dataset.get_dataset(index), **mapper_params
        )

    def close(self):
        self._ml_dataset.close()
