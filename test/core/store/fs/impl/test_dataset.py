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
from xcube.core.store.fs.impl.mldataset import MultiLevelDatasetLevelsFsDataAccessor


class DatasetZarrFsDataAccessorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = new_cube(variables=dict(temperature=279.1))
        self.temp_dir = tempfile.TemporaryDirectory()
        self.fs = fsspec.filesystem("file")
        self.data_id_v2 = f"{self.temp_dir.name}/cube_v2.zarr"
        self.data_id_v3 = f"{self.temp_dir.name}/cube_v3.zarr"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_write_and_open_data_with_cache_size(self):
        accessor = DatasetZarrFsDataAccessor()
        accessor.write_data(self.dataset, self.data_id_v2, fs=self.fs, root=None)
        opened = accessor.open_data(
            self.data_id_v2, fs=self.fs, root=None, cache_size=2**20
        )
        self.assertIsInstance(opened, xr.Dataset)
        xr.testing.assert_equal(self.dataset, opened)

    def test_write_and_open_data_zarr_v3(self):
        accessor = DatasetZarrFsDataAccessor()
        accessor.write_data(
            self.dataset, self.data_id_v3, fs=self.fs, root=None, zarr_format=3
        )
        opened = accessor.open_data(self.data_id_v3, fs=self.fs, root=None)
        self.assertIsInstance(opened, xr.Dataset)
        xr.testing.assert_equal(self.dataset, opened)

    def test_write_data_params_schema_exposes_zarr_format(self):
        accessor = DatasetZarrFsDataAccessor()
        schema = accessor.get_write_data_params_schema().to_dict()

        self.assertEqual([2, 3], schema["properties"]["zarr_format"]["enum"])
        self.assertEqual(2, schema["properties"]["zarr_format"]["default"])


class DatasetNetcdfFsDataAccessorTest(unittest.TestCase):
    @patch("xcube.core.store.fs.impl.dataset.is_https_fs", return_value=True)
    @patch("xcube.core.store.fs.impl.dataset.xr.open_dataset")
    def test_open_data_https_uses_bytes_mode(
        self, mock_open_dataset, _mock_is_https_fs
    ):
        mock_open_dataset.return_value = xr.Dataset()

        accessor = DatasetNetcdfFsDataAccessor()
        opened = accessor.open_data(
            "root.example/sample.nc", fs=fsspec.filesystem("memory"), root=None
        )

        mock_open_dataset.assert_called_once_with(
            "https://root.example/sample.nc#mode=bytes", engine="netcdf4"
        )
        self.assertIsInstance(opened, xr.Dataset)


class MultiLevelDatasetLevelsFsDataAccessorTest(unittest.TestCase):
    def test_write_data_params_schema_exposes_zarr_format(self):
        accessor = MultiLevelDatasetLevelsFsDataAccessor()
        schema = accessor.get_write_data_params_schema().to_dict()

        self.assertEqual([2, 3], schema["properties"]["zarr_format"]["enum"])
        self.assertEqual(2, schema["properties"]["zarr_format"]["default"])
