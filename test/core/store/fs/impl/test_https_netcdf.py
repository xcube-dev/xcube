#  Copyright (c) 2018-2025 by xcube team and contributors
#  Permissions are hereby granted under the terms of the MIT License:
#  https://opensource.org/licenses/MIT.

import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

from xcube.core.store import new_data_store


class HttpsNetcdfTest(unittest.TestCase):
    """
    This class tests the access of a NetCDF file from a remote HTTPS server.
    """

    @patch("xarray.open_dataset")
    def test_open_netcdf_https(self, mock_open_dataset):
        # set-up mock
        mock_data = {
            "temperature": (("time", "x", "y"), np.random.rand(5, 5, 5)),
            "precipitation": (("time", "x", "y"), np.random.rand(5, 5, 5)),
        }
        mock_ds = xr.Dataset(mock_data)
        mock_open_dataset.return_value = mock_ds

        fs_path = "mockfile.nc"
        store = new_data_store("https", root="root.de")
        ds = store.open_data(fs_path)

        mock_open_dataset.assert_called_once_with(
            "https://root.de/mockfile.nc#mode=bytes", engine="netcdf4"
        )
        self.assertTrue("temperature" in ds)
        self.assertTrue("precipitation" in ds)
        self.assertEqual(ds["temperature"].shape, (5, 5, 5))
        self.assertEqual(ds["precipitation"].shape, (5, 5, 5))
