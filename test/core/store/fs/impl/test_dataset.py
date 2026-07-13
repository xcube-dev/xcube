# Copyright (c) 2018-2026 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import tempfile
import unittest
from unittest.mock import patch

import fsspec
import xarray as xr

from xcube.core.new import new_cube
from xcube.core.store.fs.impl.dataset import (
    DatasetNetcdfFsDataAccessor,
    DatasetZarrFsDataAccessor,
)


class DatasetZarrFsDataAccessorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = new_cube(variables=dict(temperature=279.1))
        self.temp_dir = tempfile.TemporaryDirectory()
        self.fs = fsspec.filesystem("file")
        self.data_id = f"{self.temp_dir.name}/cube.zarr"
        self.dataset.to_zarr(self.data_id, zarr_version=2)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_open_data_with_cache_size(self):
        accessor = DatasetZarrFsDataAccessor()
        opened = accessor.open_data(self.data_id, fs=self.fs, root=None, cache_size=2**20)
        self.assertIsInstance(opened, xr.Dataset)
        self.assertIn("temperature", opened)

    @patch("xcube.core.store.fs.impl.dataset.xr.open_dataset")
    def test_open_data_with_non_zarr_engine(self, mock_open_dataset):
        mock_open_dataset.return_value = xr.Dataset()

        accessor = DatasetZarrFsDataAccessor()
        opened = accessor.open_data(self.data_id, fs=self.fs, root=None, engine="scipy")

        mock_open_dataset.assert_called_once()
        self.assertIsInstance(opened, xr.Dataset)


class DatasetNetcdfFsDataAccessorTest(unittest.TestCase):
    @patch("xcube.core.store.fs.impl.dataset.is_https_fs", return_value=True)
    @patch("xcube.core.store.fs.impl.dataset.xr.open_dataset")
    def test_open_data_https_uses_bytes_mode(self, mock_open_dataset, _mock_is_https_fs):
        mock_open_dataset.return_value = xr.Dataset()

        accessor = DatasetNetcdfFsDataAccessor()
        opened = accessor.open_data(
            "root.example/sample.nc", fs=fsspec.filesystem("memory"), root=None
        )

        mock_open_dataset.assert_called_once_with(
            "https://root.example/sample.nc#mode=bytes", engine="netcdf4"
        )
        self.assertIsInstance(opened, xr.Dataset)
