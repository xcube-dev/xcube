# Copyright (c) 2018-2026 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import tempfile
import unittest
from unittest.mock import MagicMock, patch

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

    def test_write_data_params_schema_exposes_zarr_format(self):
        accessor = DatasetZarrFsDataAccessor()
        schema = accessor.get_write_data_params_schema().to_dict()

        self.assertEqual(
            [2, 3], schema["properties"]["zarr_format"]["enum"]
        )
        self.assertEqual(
            2, schema["properties"]["zarr_format"]["default"]
        )

    @patch("xcube.core.store.fs.impl.dataset.is_local_fs", return_value=False)
    def test_write_data_uses_default_zarr_format_2(self, _mock_is_local_fs):
        accessor = DatasetZarrFsDataAccessor()
        dataset = MagicMock()
        fs = fsspec.filesystem("memory")

        accessor.write_data(dataset, "cube.zarr", fs=fs, root=None)

        self.assertEqual(2, dataset.to_zarr.call_args.kwargs["zarr_format"])

    @patch("xcube.core.store.fs.impl.dataset.is_local_fs", return_value=False)
    def test_write_data_passes_through_zarr_format_3(self, _mock_is_local_fs):
        accessor = DatasetZarrFsDataAccessor()
        dataset = MagicMock()
        fs = fsspec.filesystem("memory")

        accessor.write_data(dataset, "cube.zarr", fs=fs, root=None, zarr_format=3)

        self.assertEqual(3, dataset.to_zarr.call_args.kwargs["zarr_format"])


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


class MultiLevelDatasetLevelsFsDataAccessorTest(unittest.TestCase):
    def test_write_data_params_schema_exposes_zarr_format(self):
        accessor = MultiLevelDatasetLevelsFsDataAccessor()
        schema = accessor.get_write_data_params_schema().to_dict()

        self.assertEqual(
            [2, 3], schema["properties"]["zarr_format"]["enum"]
        )
        self.assertEqual(
            2, schema["properties"]["zarr_format"]["default"]
        )
