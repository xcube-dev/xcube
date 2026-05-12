# Copyright (c) 2018-2026 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import shutil
import os
import tempfile
from collections.abc import MutableMapping
from typing import Dict, Tuple, Union

import numpy as np
import xarray as xr
import zarr

from xcube.core.chunk import chunk_dataset

DEFAULT_TIME_EPS = np.array(1000 * 1000, dtype="timedelta64[ns]")


def find_time_slice(
    store: Union[str, MutableMapping],
    time_stamp: Union[np.datetime64, np.ndarray],
    time_eps: np.timedelta64 = DEFAULT_TIME_EPS,
) -> tuple[int, str]:
    """Find time index and update mode for *time_stamp* in
    Zarr dataset given by *store*.

    Args:
        store: A zarr store.
        time_stamp: Time stamp to find index for.
        time_eps: Time epsilon for equality comparison,
            defaults to 1 millisecond.
    Returns:
        A tuple (time_index, 'insert') or (time_index, 'replace')
        if an index was found, (-1, 'create') or (-1, 'append')
        otherwise.
    """
    try:
        cube = xr.open_zarr(store)
    except (FileNotFoundError, ValueError):
        # FileNotFoundError is raised as by Zarr since 2.13,
        # before GroupNotFoundError (extends ValueError) was raised.
        # Keep ValueError for backward compatibility.
        try:
            cube = xr.open_dataset(store)
        except (FileNotFoundError, ValueError):
            # If the zarr directory does not exist, open_dataset raises a
            # FileNotFoundError (with xarray <= 0.17.0) or a ValueError
            # (with xarray 0.18.0).
            return -1, "create"

    # TODO (forman): optimise following naive search by bi-sectioning or so
    for i in range(cube.time.size):
        time = cube.time[i]
        if abs(time_stamp - time) < time_eps:
            return i, "replace"
        if time_stamp < time:
            return i, "insert"

    return -1, "append"


def append_time_slice(
    store: Union[str, MutableMapping],
    time_slice: xr.Dataset,
    chunk_sizes: dict[str, int] = None,
):
    """Append time slice to existing zarr dataset.

    Args:
        store: A zarr store.
        time_slice: Time slice to insert
        chunk_sizes: desired chunk sizes
    """
    if chunk_sizes:
        time_slice = chunk_dataset(time_slice, chunk_sizes, format_name="zarr")

    ds = zarr.open_group(store, mode="r")

    time_slice = time_slice.copy()
    time_slice.attrs.update(ds.attrs)

    # remove legacy attribute conflict
    time_slice.attrs.pop("coordinates", None)

    time_slice.to_zarr(
        store,
        mode="a",
        append_dim="time",
        consolidated=True,
    )


def insert_time_slice(
    store: Union[str, MutableMapping],
    insert_index: int,
    time_slice: xr.Dataset,
    chunk_sizes: dict[str, int] = None,
):
    """Insert time slice into existing zarr dataset.

    Args:
        store: A zarr store.
        insert_index: Time index
        time_slice: Time slice to insert
        chunk_sizes: desired chunk sizes
    """
    update_time_slice(
        store, insert_index, time_slice, "insert", chunk_sizes=chunk_sizes
    )


def replace_time_slice(
    store: Union[str, MutableMapping],
    insert_index: int,
    time_slice: xr.Dataset,
    chunk_sizes: dict[str, int] = None,
):
    """Replace time slice in existing zarr dataset.

    Args:
        store: A zarr store.
        insert_index: Time index
        time_slice: Time slice to insert
        chunk_sizes: desired chunk sizes
    """
    update_time_slice(
        store, insert_index, time_slice, "replace", chunk_sizes=chunk_sizes
    )


def update_time_slice(
    store: Union[str, MutableMapping],
    insert_index: int,
    time_slice: xr.Dataset,
    mode: str,
    chunk_sizes: dict[str, int] = None,
):
    """Update existing zarr dataset by new time slice.

    Args:
        store: A zarr store.
        insert_index: Time index
        time_slice: Time slice to insert
        mode: Update mode, 'insert' or 'replace'
        chunk_sizes: desired chunk sizes
    """
    if mode not in ("insert", "replace"):
        raise ValueError(f"illegal mode value: {mode!r}")

    insert_mode = mode == "insert"

    if chunk_sizes:
        time_slice = chunk_dataset(time_slice, chunk_sizes, format_name="zarr")

    cube = xr.open_zarr(store)

    # --- split dataset into before / after ---
    if insert_mode:
        before = cube.isel(time=slice(0, insert_index))
        after = cube.isel(time=slice(insert_index, None))
        new_ds = xr.concat([before, time_slice, after], dim="time")
    else:
        before = cube.isel(time=slice(0, insert_index))
        after = cube.isel(time=slice(insert_index + 1, None))
        new_ds = xr.concat([before, time_slice, after], dim="time")
    new_ds = xr.unify_chunks(new_ds)[0]

    # preserve global attrs
    new_ds.attrs.update(cube.attrs)

    tmp = tempfile.mkdtemp(prefix="xcube-timeslice-")
    new_ds.to_zarr(tmp, mode="w", consolidated=True)
    if os.path.isdir(store):
        shutil.rmtree(store)
    shutil.move(tmp, store)
