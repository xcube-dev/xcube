# Copyright (c) 2018-2026 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import uuid

import numpy as np
import pytest
import xarray as xr
import zarr

from xcube.core.store import new_data_store


def _new_dataset() -> xr.Dataset:
    return xr.Dataset(
        data_vars={"a": (("time", "x"), np.array([[1, 2], [3, 4]], dtype=np.int16))},
        coords={"time": [0, 1], "x": [10, 20]},
    )


def _new_memory_store():
    root = f"xcube-zarr-format-{uuid.uuid4().hex}"
    return new_data_store("memory", root=root), root


def test_memory_store_writes_zarr_v2_by_default():
    store, root = _new_memory_store()
    data_id = "default.zarr"

    store.write_data(_new_dataset(), data_id, replace=True)

    fs_path = f"{root}/{data_id}"
    assert store.fs.exists(f"{fs_path}/.zgroup")
    assert not store.fs.exists(f"{fs_path}/zarr.json")

    actual = store.open_data(data_id)
    np.testing.assert_array_equal(actual.a.values, [[1, 2], [3, 4]])


@pytest.mark.skipif(
    int(zarr.__version__.split(".", maxsplit=1)[0]) < 3,
    reason="Zarr format 3 writing requires zarr-python 3",
)
def test_memory_store_writes_zarr_v3_when_requested():
    store, root = _new_memory_store()
    data_id = "requested-v3.zarr"

    store.write_data(_new_dataset(), data_id, replace=True, zarr_format=3)

    fs_path = f"{root}/{data_id}"
    assert store.fs.exists(f"{fs_path}/zarr.json")
    assert not store.fs.exists(f"{fs_path}/.zgroup")

    actual = store.open_data(data_id, zarr_format=3)
    np.testing.assert_array_equal(actual.a.values, [[1, 2], [3, 4]])
