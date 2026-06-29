# Copyright (c) 2018-2026 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from collections.abc import Sequence

import numpy as np
import xarray as xr

from xcube.core.zarrcompat import (
    consolidate_zarr_metadata,
    detect_zarr_format,
    is_zarr_path,
    open_zarr_group,
)


def unchunk_dataset(
    dataset_path: str, var_names: Sequence[str] = None, coords_only: bool = False
):
    """Unchunk dataset variables in-place.

    Args:
        dataset_path: Path to ZARR dataset directory.
        var_names: Optional list of variable names.
        coords_only: Un-chunk coordinate variables only.
    """

    if not is_zarr_path(dataset_path):
        raise ValueError(f"{dataset_path!r} is not a valid Zarr directory")

    with xr.open_zarr(dataset_path) as dataset:
        if var_names is None:
            if coords_only:
                var_names = list(dataset.coords)
            else:
                var_names = list(dataset.variables)
        else:
            for var_name in var_names:
                if coords_only:
                    if var_name not in dataset.coords:
                        raise ValueError(
                            f"variable {var_name!r} is not a coordinate variable in {dataset_path!r}"
                        )
                else:
                    if var_name not in dataset.variables:
                        raise ValueError(
                            f"variable {var_name!r} is not a variable in {dataset_path!r}"
                        )

    _unchunk_vars(dataset_path, var_names)


def _unchunk_vars(dataset_path: str, var_names: list[str]):
    zarr_format = detect_zarr_format(dataset_path)
    root = open_zarr_group(dataset_path, mode="a", zarr_format=zarr_format)
    for var_name in var_names:
        var_array = root[var_name]

        if tuple(var_array.shape) == tuple(var_array.chunks):
            continue

        # TODO (forman): Fully loading data is inefficient and dangerous for large arrays.
        data = np.array(var_array)
        attributes = _get_attrs(var_array)
        fill_value = getattr(var_array, "fill_value", None)
        metadata = getattr(var_array, "metadata", None)
        dimension_names = getattr(metadata, "dimension_names", None)

        del root[var_name]
        kwargs = dict(
            name=var_name,
            data=data,
            chunks=data.shape,
            fill_value=fill_value,
        )
        if dimension_names is not None:
            kwargs["dimension_names"] = dimension_names

        if hasattr(root, "create_array"):
            new_array = root.create_array(**kwargs)
        else:
            kwargs.pop("name")
            new_array = root.create_dataset(var_name, **kwargs)

        new_array.attrs.update(attributes)

    consolidate_zarr_metadata(dataset_path, zarr_format=zarr_format)


def _get_attrs(var_array) -> dict:
    attrs = var_array.attrs
    if hasattr(attrs, "asdict"):
        return attrs.asdict()
    return dict(attrs)
