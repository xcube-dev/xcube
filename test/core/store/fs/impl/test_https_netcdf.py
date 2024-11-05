# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import os.path
import unittest

import fsspec
import rasterio as rio
import rioxarray
import s3fs
import xarray
import xarray as xr

from xcube.core.store import new_data_store


class HttpsNetcdfTest(unittest.TestCase):
    """
    This class tests a lazy access of a NetCDF file from a remote HTTPS server.
    """

    def test_open_netcdf_https(self):
        """This test loads GAMIv2-0_2010-2020_100m.nc (217GB), which is available via
        https://datapub.gfz-potsdam.de/download/10.5880.GFZ.1.4.2023.006-VEnuo/
        """
        fs_path = "download/10.5880.GFZ.1.4.2023.006-VEnuo/GAMIv2-0_2010-2020_100m.nc"
        store = new_data_store("https", root="datapub.gfz-potsdam.de")
        ds = store.open_data(fs_path, chunks={})
        self.assertIsInstance(ds, xr.Dataset)
        self.assertEqual(
            {"members": 20, "latitude": 202500, "longitude": 405000, "time": 2},
            ds.sizes,
        )
        self.assertEqual(
            [1, 7789, 15577, 1],
            [
                ds.chunksizes["members"][0],
                ds.chunksizes["latitude"][0],
                ds.chunksizes["longitude"][0],
                ds.chunksizes["time"][0],
            ],
        )
