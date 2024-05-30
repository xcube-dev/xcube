# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import collections.abc
from typing import Union, Set, Dict
from collections.abc import Iterator

import xarray as xr

from xcube.core.mldataset import MultiLevelDataset
from ..datasets.context import DatasetsContext
from ...util.assertions import assert_instance

_LEVELS_EXT = ".levels"
_ZARR_EXT = ".zarr"


class DatasetsMapping(collections.abc.Mapping):
    """Represents the given *datasets_ctx* as a mapping from
    dataset identifier to dataset, it can
    be passed to class:ObjectStorage.

    This is the applied Adapter design pattern to make
    class:DatasetsContext compatible with the mapping argument for
    class:ObjectStorage.

    The original identifiers will be renamed in case their suffixes
    do not match the desired bucket type, so the bucket contents are
    more user-friendly. If *is_multi_level* is True, the new names
    will always have ".levels" suffix, otherwise the ".zarr" suffix.

    Args:
        datasets_ctx: The datasets' context
        is_multi_level: Whether this is a multi-level datasets' object
            storage
    """

    def __init__(self, datasets_ctx: DatasetsContext, is_multi_level: bool = False):
        assert_instance(datasets_ctx, DatasetsContext, name="datasets_ctx")
        assert_instance(is_multi_level, bool, name="is_multi_level")
        self._datasets_ctx = datasets_ctx
        self._is_multi_level = is_multi_level

    @property
    def _s3_names(self):
        return self._get_s3_names(self._datasets_ctx, self._is_multi_level)

    @staticmethod
    def _get_s3_names(
        datasets_ctx: DatasetsContext, is_multi_level: bool
    ) -> dict[str, str]:
        """Generate user-friendly S3 names for dataset identifiers.
        If *is_multi_level* is True, S3 names will be forced to have
        ".levels" suffix, otherwise the ".zarr" suffix.
        """
        ds_ids = [c["Identifier"] for c in datasets_ctx.get_dataset_configs()]
        all_ids = set(ds_ids)

        s3_names = {}
        for ds_id in ds_ids:
            s3_base, s3_ext = _split_base_ext(ds_id)
            if is_multi_level:
                s3_name = _replace_ext(s3_base, s3_ext, _ZARR_EXT, _LEVELS_EXT, all_ids)
            else:
                s3_name = _replace_ext(s3_base, s3_ext, _LEVELS_EXT, _ZARR_EXT, all_ids)
            s3_names[s3_name] = ds_id
        return s3_names

    def __len__(self) -> int:
        return len(self._s3_names)

    def __iter__(self) -> Iterator[str]:
        return iter(self._s3_names)

    def __contains__(self, s3_name: str) -> bool:
        """Check if *dataset_id* is a valid dataset.
        Overridden to avoid a call to __getitem__(),
        which will open the dataset (or raise ApiError!),
        but we want this to happen for direct __getitem__()
        calls only.
        """
        return s3_name in self._s3_names

    def __getitem__(self, s3_name: str) -> Union[xr.Dataset, MultiLevelDataset]:
        """Get or open the dataset given by *dataset_id*."""
        dataset_id = self._s3_names[s3_name]
        # Will raise ApiError
        if self._is_multi_level:
            return self._datasets_ctx.get_ml_dataset(dataset_id)
        else:
            return self._datasets_ctx.get_dataset(dataset_id)


def _split_base_ext(identifier: str):
    base_ext = identifier.rsplit(".", maxsplit=1)
    if len(base_ext) == 2:
        return base_ext[0], "." + base_ext[1]
    else:
        return identifier, ""


def _replace_ext(
    base_name: str, old_ext: str, trigger_ext: str, new_ext: str, all_ids: set[str]
) -> str:
    if old_ext == new_ext or (
        old_ext == trigger_ext and (base_name + new_ext) not in all_ids
    ):
        return base_name + new_ext
    return base_name + old_ext + new_ext
