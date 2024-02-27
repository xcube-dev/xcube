import glob
import json
from pathlib import Path
from typing import List
import unittest

import fsspec
import numpy as np
import xarray as xr

from xcube.core.store import DataStore
from xcube.core.store import MutableDataStore
from xcube.core.store import new_data_store

try:
    from kerchunk.netCDF3 import NetCDF3ToZarr
    from kerchunk.combine import MultiZarrToZarr
except ImportError:
    NetCDF3ToZarr = None
    MultiZarrToZarr = None


@unittest.skipIf(NetCDF3ToZarr is None, reason="kerchunk not installed")
class ReferenceFsDataStoresTest(unittest.TestCase):
    reference_files: List[str] = []

    @classmethod
    def setUpClass(cls):
        data_dir = (Path(__file__).parent / "../../../core/gen/inputdata").resolve()
        data_file_paths = sorted(glob.glob(str(data_dir / "*-IFR-L4_GHRSST*.nc")))
        reference_file_paths = [data_file_path + ".json" for data_file_path in
                                data_file_paths]
        for data_file_path, reference_file_path in zip(data_file_paths,
                                                       reference_file_paths):
            h5chunks = NetCDF3ToZarr(data_file_path)
            with open(reference_file_path, mode="w") as json_stream:
                json.dump(h5chunks.translate(), json_stream)

        cube_ref = MultiZarrToZarr(reference_file_paths,
                                   remote_protocol="file",
                                   concat_dims=["time"],
                                   identical_dims=["lat", "lon"])
        cube_ref_path = str(data_dir / "sst-cube.json")
        with open(cube_ref_path, "w") as f:
            json.dump(cube_ref.translate(), f)

        cls.reference_files = [cube_ref_path] + reference_file_paths

    @classmethod
    def tearDownClass(cls):
        fs = fsspec.filesystem("file")
        for reference_file in cls.reference_files:
            fs.delete(reference_file)

    def test_store(self):
        store = new_data_store("reference", ref_paths=self.reference_files)
        self.assertIsInstance(store, DataStore)
        self.assertNotIsInstance(store, MutableDataStore)
        self.assertEqual(
            [
                "sst-cube",
                "20170101-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc",
                "20170102-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc",
                "20170103-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc",
            ],
            list(store.get_data_ids())
        )

    def test_reference_filesystem(self):
        cube_ref_path = self.reference_files[0]
        fs = fsspec.filesystem("reference",
                               fo=cube_ref_path,
                               remote_protocol="file")
        self.assertEqual(
            {
                '.zattrs',
                '.zgroup',
                'time',
                'lat',
                'lon',
                'analysed_sst',
                'analysis_error',
                'mask',
                'sea_ice_fraction'
            },
            set(fs.ls("", detail=False))
        )
        zarr_store = fs.get_mapper()
        cube = xr.open_zarr(zarr_store, consolidated=False)
        self.assert_cube_ok(cube)

    def test_xr_open_dataset(self):
        cube_ref_path = self.reference_files[0]
        cube = xr.open_dataset("reference://",
                               engine="zarr",
                               backend_kwargs=dict(
                                   consolidated=False,
                                   storage_options=dict(
                                       fo=cube_ref_path,
                                       remote_protocol="file"
                                   )
                               ))
        self.assert_cube_ok(cube)

    def assert_cube_ok(self, cube: xr.Dataset):
        self.assertEqual({'time': 3, 'lat': 1350, 'lon': 1600}, cube.sizes)
        self.assertEqual({'lon', 'time', 'lat'}, set(cube.coords))
        self.assertEqual({'analysis_error', 'mask', 'analysed_sst', 'sea_ice_fraction'},
                         set(cube.data_vars))
        sst_ts = cube.isel(lat=0, lon=0).compute()
        np.testing.assert_array_equal(sst_ts.analysed_sst.values,
                                      np.array([290.02, 289.94, 289.89]))
