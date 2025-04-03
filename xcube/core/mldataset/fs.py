# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import collections.abc
import json
import math
import pathlib
import warnings
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Any, Dict, List, Optional, Union

import fsspec
import fsspec.core
import numpy as np
import xarray as xr
import zarr

# noinspection PyUnresolvedReferences
import xcube.core.zarrstore
from xcube.core.gridmapping import GridMapping
from xcube.core.subsampling import AggMethod, AggMethods
from xcube.util.assertions import assert_instance
from xcube.util.fspath import get_fs_path_class, resolve_path
from xcube.util.types import ScalarOrPair, normalize_scalar_or_pair

from .abc import MultiLevelDataset
from .base import BaseMultiLevelDataset
from .lazy import LazyMultiLevelDataset

LEVELS_FORMAT_VERSION = "1.0"


class FsMultiLevelDataset(LazyMultiLevelDataset):
    _MIN_CACHE_SIZE = 1024 * 1024  # 1 MiB

    def __init__(
        self,
        path: str,
        fs: Optional[fsspec.AbstractFileSystem] = None,
        fs_root: Optional[str] = None,
        fs_kwargs: Optional[Mapping[str, Any]] = None,
        cache_size: Optional[int] = None,
        consolidate: Optional[bool] = None,
        **zarr_kwargs,
    ):
        if fs is None:
            fs, path = fsspec.core.url_to_fs(path, **(fs_kwargs or {}))
        assert_instance(fs, fsspec.AbstractFileSystem, name="fs")
        assert_instance(path, str, name="data_id")
        # TODO: Setting ds_id=path is likely the root cause for
        #   https://github.com/xcube-dev/xcube/issues/1007.
        #   Then, the actual fix is to replace slashes by dashes or underscores.
        #   But this requires deeper investigation and more test cases.
        #   A quick fix has been applied in `xcube.webapi.viewer.Viewer.add_dataset()`
        #   implementation. forman, 2024-06-07
        super().__init__(ds_id=path)
        self._path = path
        self._fs = fs
        self._fs_root = fs_root
        self._cache_size = cache_size
        self._consolidate = consolidate
        self._zarr_kwargs = zarr_kwargs
        self._path_class = get_fs_path_class(fs)

    @property
    def path(self) -> str:
        return self._path

    @property
    def fs(self) -> fsspec.AbstractFileSystem:
        return self._fs

    @property
    def cache_size(self) -> Optional[int]:
        return self._cache_size

    @cached_property
    def size_weights(self) -> np.ndarray:
        """Size weights are used to distribute the cache size
        over the levels.
        """
        return self.compute_size_weights(self.num_levels)

    @cached_property
    def tile_size(self) -> Optional[Sequence[int]]:
        spec = self._levels_spec
        return spec.get("tile_size")

    @cached_property
    def use_saved_levels(self) -> Optional[bool]:
        spec = self._levels_spec
        return spec.get("use_saved_levels")

    @cached_property
    def base_dataset_path(self) -> Optional[str]:
        spec = self._levels_spec
        return spec.get("base_dataset_path")

    @cached_property
    def base_dataset_params(self) -> Optional[dict[str, Any]]:
        spec = self._levels_spec
        return spec.get("base_dataset_params")

    @cached_property
    def agg_methods(self) -> Optional[Mapping[str, AggMethod]]:
        spec = self._levels_spec
        return spec.get("agg_methods")

    @cached_property
    def _levels_spec(self) -> Mapping[str, Any]:
        path = f"{self._path}/.zlevels"
        spec = {}
        if self.fs.exists(path):
            with self.fs.open(f"{self._path}/.zlevels") as fp:
                spec = json.load(fp)
            if not isinstance(spec, collections.abc.Mapping):
                raise TypeError("Unexpected .zlevels file. Must be a JSON object.")
        # TODO (forman): validate JSON object
        return spec

    def _get_dataset_lazily(self, index: int, parameters) -> xr.Dataset:
        cache_size = self._cache_size

        fs = self._fs

        open_params = dict(self._zarr_kwargs)
        base_dataset_open_params = None

        ds_path = self._get_path(self._path)
        link_path = ds_path / f"{index}.link"
        if fs.isfile(str(link_path)):
            # If file "{index}.link" exists, we have a link to
            # a level Zarr and open this instead,
            with fs.open(str(link_path), "r") as fp:
                level_path = self._get_path(fp.read())
            if not level_path.is_absolute() and not self._is_path_relative_to_path(
                level_path, ds_path
            ):
                level_path = resolve_path(ds_path / level_path)
            base_dataset_open_params = self.base_dataset_params
        else:
            # Nominal "{index}.zarr" must exist
            level_path = ds_path / f"{index}.zarr"

        if isinstance(base_dataset_open_params, dict):
            # TODO: complete logic here
            engine = base_dataset_open_params.pop("engine", "zarr")

        level_zarr_store = fs.get_mapper(str(level_path))

        consolidated = (
            self._consolidate
            if self._consolidate is not None
            else (".zmetadata" in level_zarr_store)
        )

        if isinstance(cache_size, int) and cache_size >= self._MIN_CACHE_SIZE:
            # compute cache size for level weighted by
            # size in pixels for each level
            cache_size = math.ceil(self.size_weights[index] * cache_size)
            if cache_size >= self._MIN_CACHE_SIZE:
                level_zarr_store = zarr.LRUStoreCache(
                    level_zarr_store, max_size=cache_size
                )

        try:
            level_dataset = xr.open_zarr(
                level_zarr_store, consolidated=consolidated, **self._zarr_kwargs
            )
        except ValueError as e:
            raise FsMultiLevelDatasetError(
                f"Failed to open dataset {level_path!r}: {e}"
            ) from e

        level_dataset.zarr_store.set(level_zarr_store)
        return level_dataset

    @staticmethod
    def _is_path_relative_to_path(level_path, ds_path):
        if hasattr(level_path, "is_relative_to"):
            # Python >=3.9
            return level_path.is_relative_to(ds_path)
        try:
            # Python <3.9
            level_path.relative_to(ds_path)
            return True
        except ValueError:
            return False

    @classmethod
    def compute_size_weights(cls, num_levels: int) -> np.ndarray:
        weights = (2 ** np.arange(0, num_levels, dtype=np.float64)) ** 2
        return weights[::-1] / np.sum(weights)

    def _get_num_levels_lazily(self) -> int:
        spec = self._levels_spec
        num_levels = spec.get("num_levels")
        levels = self._get_levels()
        if num_levels is None:
            num_levels = len(levels)
        expected_levels = list(range(num_levels))
        for level in expected_levels:
            if level != levels[level]:
                raise FsMultiLevelDatasetError(
                    f"Inconsistent"
                    f" multi-level dataset {self.ds_id!r},"
                    f" expected levels {expected_levels!r}"
                    f" found {levels!r}"
                )
        return num_levels

    def _get_levels(self) -> list[int]:
        levels = []
        paths = [
            self._get_path(entry["name"])
            for entry in self._fs.listdir(self._path, detail=True)
        ]
        for path in paths:
            # No ext, i.e. dir_name = "<level>", is proposed by
            # https://github.com/zarr-developers/zarr-specs/issues/50.
            # xcube already selected dir_name = "<level>.zarr".
            basename = path.stem
            if path.stem and path.suffix in ("", ".zarr", ".link"):
                try:
                    level = int(basename)
                except ValueError:
                    continue
                levels.append(level)
        levels = sorted(levels)
        return levels

    def _get_path(self, *args) -> pathlib.PurePath:
        return self._path_class(*args)

    @classmethod
    def write_dataset(
        cls,
        dataset: Union[xr.Dataset, MultiLevelDataset],
        path: str,
        fs: Optional[fsspec.AbstractFileSystem] = None,
        fs_root: Optional[str] = None,
        fs_kwargs: Optional[Mapping[str, Any]] = None,
        replace: bool = False,
        num_levels: Optional[int] = None,
        consolidated: bool = True,
        tile_size: Optional[ScalarOrPair[int]] = None,
        use_saved_levels: bool = False,
        base_dataset_path: Optional[str] = None,
        base_dataset_params: Optional[dict[str, Any]] = None,
        agg_methods: Optional[AggMethods] = None,
        **zarr_kwargs,
    ) -> str:
        assert_instance(dataset, (xr.Dataset, MultiLevelDataset), name="dataset")

        if fs is None:
            fs, path = fsspec.core.url_to_fs(path, **(fs_kwargs or {}))
        if tile_size is not None:
            tile_size = normalize_scalar_or_pair(
                tile_size, item_type=int, name="tile_size"
            )

        assert_instance(path, str, name="path")
        assert_instance(fs, fsspec.AbstractFileSystem, name="fs")

        if isinstance(dataset, MultiLevelDataset):
            ml_dataset = dataset
            if tile_size:
                warnings.warn("tile_size is ignored for multi-level datasets")
            if agg_methods:
                warnings.warn("agg_methods is ignored for multi-level datasets")
        else:
            base_dataset: xr.Dataset = dataset
            grid_mapping = None
            if tile_size is not None:
                grid_mapping = GridMapping.from_dataset(base_dataset)
                x_name, y_name = grid_mapping.xy_dim_names
                # noinspection PyTypeChecker
                base_dataset = base_dataset.chunk(
                    {x_name: tile_size[0], y_name: tile_size[1]}
                )
                # noinspection PyTypeChecker
                grid_mapping = grid_mapping.derive(tile_size=tile_size)
            ml_dataset = BaseMultiLevelDataset(
                base_dataset,
                grid_mapping=grid_mapping,
                num_levels=num_levels,
                agg_methods=agg_methods,
            )

        if use_saved_levels:
            ml_dataset = BaseMultiLevelDataset(
                ml_dataset.get_dataset(0),
                grid_mapping=ml_dataset.grid_mapping,
                agg_methods=agg_methods,
            )

        path_class = get_fs_path_class(fs)
        data_path = path_class(path)
        fs.mkdirs(str(data_path), exist_ok=replace)

        if num_levels is None or num_levels <= 0:
            num_levels_max = ml_dataset.num_levels
        else:
            num_levels_max = min(num_levels, ml_dataset.num_levels)

        with fs.open(str(data_path / ".zlevels"), mode="w") as fp:
            levels_data: dict[str, Any] = dict(
                version=LEVELS_FORMAT_VERSION, num_levels=num_levels_max
            )
            if use_saved_levels is not None:
                levels_data.update(use_saved_levels=bool(use_saved_levels))
            if base_dataset_path:
                levels_data.update(base_dataset_path=base_dataset_path)
            if base_dataset_params is not None:
                levels_data.update(base_dataset_params=base_dataset_params)
            if tile_size is not None:
                levels_data.update(tile_size=list(tile_size))
            if hasattr(ml_dataset, "agg_methods"):
                levels_data.update(agg_methods=dict(ml_dataset.agg_methods))
            json.dump(levels_data, fp, indent=2)

        for index in range(num_levels_max):
            level_dataset = ml_dataset.get_dataset(index)
            if base_dataset_path and index == 0:
                assert_instance(fs_root, str, name="fs_root")

                # Write file "0.link" instead of copying
                # level zero dataset to "0.zarr".

                # Compute a relative base dataset path first
                base_dataset_path = path_class(fs_root, base_dataset_path)
                data_parent_path = data_path.parent
                try:
                    base_dataset_path = base_dataset_path.relative_to(data_parent_path)
                except ValueError as e:
                    raise FsMultiLevelDatasetError(
                        f"Invalid base_dataset_id: {base_dataset_path}"
                    ) from e
                base_dataset_path = ".." / base_dataset_path

                # Then write relative base dataset path into link file
                link_path = data_path / f"{index}.link"
                with fs.open(str(link_path), mode="w") as fp:
                    fp.write(base_dataset_path.as_posix())
            else:
                # Write level "{index}.zarr"
                level_path = data_path / f"{index}.zarr"
                level_zarr_store = fs.get_mapper(str(level_path), create=True)
                try:
                    level_dataset.to_zarr(
                        level_zarr_store,
                        mode="w" if replace else None,
                        consolidated=consolidated,
                        **zarr_kwargs,
                    )
                except ValueError as e:
                    # TODO: remove already written data!
                    raise FsMultiLevelDatasetError(
                        f"Failed to write dataset {path}: {e}"
                    ) from e
                if use_saved_levels:
                    level_dataset = xr.open_zarr(
                        level_zarr_store, consolidated=consolidated
                    )
                    level_dataset.zarr_store.set(level_zarr_store)
                    ml_dataset.set_dataset(index, level_dataset)

        return path


class FsMultiLevelDatasetError(ValueError):
    def __init__(self, message: str):
        super().__init__(message)
