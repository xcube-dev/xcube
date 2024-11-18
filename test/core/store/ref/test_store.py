# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import glob
import json
import unittest
from abc import ABC, abstractmethod
from pathlib import Path
import warnings

import fsspec
import pytest
import xarray as xr

from xcube.core.store import DataStore, DatasetDescriptor, DataType
from xcube.core.store import MutableDataStore
from xcube.core.store import new_data_store
from xcube.core.store.ref.schema import REF_STORE_SCHEMA
from xcube.core.store.ref.store import ReferenceDataStore

try:
    from kerchunk.netCDF3 import NetCDF3ToZarr
    from kerchunk.combine import MultiZarrToZarr

    has_kerchunk = True
except ImportError:
    NetCDF3ToZarr = None
    MultiZarrToZarr = None
    has_kerchunk = False


def create_ref_paths():
    data_dir = (Path(__file__).parent / "../../../core/gen/inputdata").resolve()
    data_file_paths = sorted(glob.glob(str(data_dir / "*-IFR-L4_GHRSST*.nc")))
    reference_file_paths = [
        data_file_path + ".json" for data_file_path in data_file_paths
    ]
    for data_file_path, reference_file_path in zip(
        data_file_paths, reference_file_paths
    ):
        h5chunks = NetCDF3ToZarr(data_file_path)
        with open(reference_file_path, mode="w") as json_stream:
            json.dump(h5chunks.translate(), json_stream)

    cube_ref = MultiZarrToZarr(
        reference_file_paths,
        remote_protocol="file",
        concat_dims=["time"],
        identical_dims=["lat", "lon"],
    )
    cube_ref_path = str(data_dir / "sst-cube.json")
    with open(cube_ref_path, "w") as f:
        json.dump(cube_ref.translate(), f)

    return [cube_ref_path] + reference_file_paths


def delete_ref_paths(ref_paths: list[str]):
    fs = fsspec.filesystem("file")
    for ref_path in ref_paths:
        fs.delete(ref_path)


# noinspection PyPep8Naming,PyUnresolvedReferences
class KerchunkMixin:
    ref_paths: list[str] = []

    @classmethod
    def setUpClass(cls):
        cls.ref_paths = create_ref_paths()

    @classmethod
    def tearDownClass(cls):
        delete_ref_paths(cls.ref_paths)

    def assert_sst_cube_ok(self, cube: xr.Dataset):
        self.assertEqual({"time": 3, "lat": 1350, "lon": 1600}, cube.sizes)
        self.assertEqual({"lon", "time", "lat"}, set(cube.coords))
        self.assertEqual(
            {"analysis_error", "mask", "analysed_sst", "sea_ice_fraction"},
            set(cube.data_vars),
        )
        sst_ts = cube.isel(lat=0, lon=0).compute()
        sst_data = sst_ts.analysed_sst.values
        self.assertAlmostEqual(290.02, sst_data[0], 2)
        self.assertAlmostEqual(289.94, sst_data[1], 2)
        self.assertAlmostEqual(289.89, sst_data[2], 2)


# noinspection PyUnresolvedReferences
class ReferenceDataStoreTestBase(KerchunkMixin, ABC):
    @abstractmethod
    def get_store(self) -> DataStore:
        pass

    def test_has_store(self):
        store = self.get_store()
        self.assertIsInstance(store, DataStore)
        self.assertNotIsInstance(store, MutableDataStore)

    def test_get_data_store_params_schema(self):
        store = self.get_store()
        self.assertEqual(
            REF_STORE_SCHEMA.to_dict(), store.get_data_store_params_schema().to_dict()
        )

    def test_list_data_ids(self):
        store = self.get_store()
        self.assertEqual(
            [
                "sst-cube",
                "20170101-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc",
                "20170102-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc",
                "20170103-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc",
            ],
            store.list_data_ids(),
        )

    def test_get_data_opener_ids(self):
        store = self.get_store()
        self.assertEqual(("dataset:zarr:reference",), store.get_data_opener_ids())

    def test_get_data_types(self):
        store = self.get_store()
        self.assertEqual(("dataset",), store.get_data_types())
        self.assertEqual(("dataset",), store.get_data_types_for_data("sst-cube"))

    def test_has_data(self):
        store = self.get_store()
        self.assertEqual(True, store.has_data("sst-cube"))
        self.assertEqual(False, store.has_data("lst-cube"))

    def test_describe_data(self):
        store = self.get_store()
        descriptor = store.describe_data("sst-cube")
        self.assertIsInstance(descriptor, DatasetDescriptor)
        self.assertEqual("sst-cube", descriptor.data_id)
        self.assertIsInstance(descriptor.data_type, DataType)
        self.assertIs(xr.Dataset, descriptor.data_type.dtype)
        self.assertIsInstance(descriptor.bbox, tuple)
        self.assertIsNone(descriptor.spatial_res)  # ?
        self.assertIsInstance(descriptor.dims, dict)
        self.assertIsInstance(descriptor.coords, dict)
        self.assertIsInstance(descriptor.data_vars, dict)
        self.assertIsInstance(descriptor.attrs, dict)

    def test_open_data(self):
        store = self.get_store()
        sst_cube = store.open_data("sst-cube")
        self.assert_sst_cube_ok(sst_cube)
        sst_cube = store.open_data("sst-cube", data_type="dataset")
        self.assert_sst_cube_ok(sst_cube)
        with warnings.catch_warnings(record=True) as w:
            sst_cube = store.open_data("sst-cube", data_type="mldataset")
            self.assertEqual(1, len(w))
            self.assertEqual(w[0].category, UserWarning)
            self.assertEqual(
                (
                    "ReferenceDataStore can only represent "
                    "the data resource as xr.Dataset."
                ),
                w[0].message.args[0],
            )
        self.assert_sst_cube_ok(sst_cube)

    def test_get_search_params_schema(self):
        store = self.get_store()
        # We do not have search parameters yet
        self.assertEqual(
            {"type": "object", "properties": {}},
            store.get_search_params_schema().to_dict(),
        )

    def test_search_data(self):
        store = self.get_store()
        search_results = list(store.search_data())
        self.assertEqual(4, len(search_results))
        for descriptor, data_id in zip(search_results, store.get_data_ids()):
            self.assertIsInstance(descriptor, DatasetDescriptor)
            self.assertEqual(data_id, descriptor.data_id)
            self.assertIsInstance(descriptor.data_type, DataType)
            self.assertIs(xr.Dataset, descriptor.data_type.dtype)
            self.assertIsInstance(descriptor.bbox, tuple)
            self.assertIsNone(descriptor.spatial_res)  # ?
            self.assertIsInstance(descriptor.dims, dict)
            self.assertIsInstance(descriptor.coords, dict)
            self.assertIsInstance(descriptor.data_vars, dict)
            self.assertIsInstance(descriptor.attrs, dict)


@unittest.skipUnless(has_kerchunk, reason="kerchunk not installed")
class ReferenceDataStorePathsTest(ReferenceDataStoreTestBase, unittest.TestCase):
    def get_store(self) -> DataStore:
        return new_data_store("reference", refs=self.ref_paths)


@unittest.skipUnless(has_kerchunk, reason="kerchunk not installed")
class ReferenceDataStoreDictsTest(ReferenceDataStoreTestBase, unittest.TestCase):
    def get_store(self) -> DataStore:
        store = new_data_store("reference", refs=self.ref_paths)
        refs = [
            dict(
                ref_path=ref_path,
                data_descriptor=store.describe_data(data_id).to_dict(),
            )
            for ref_path, data_id in zip(self.ref_paths, store.get_data_ids())
        ]
        return new_data_store("reference", refs=refs)


class NormalizeRefTest(unittest.TestCase):
    normalize_ref = ReferenceDataStore._normalize_ref

    def test_normalize_str(self):
        self.assertEqual(
            (
                "sst-cube",
                {
                    "ref_path": "https://myrefs.com/sst-cube.json",
                    "data_descriptor": None,
                },
            ),
            self.normalize_ref("https://myrefs.com/sst-cube.json"),
        )

    def test_normalize_dict_with_ref_path(self):
        self.assertEqual(
            (
                "sst-bert",
                {
                    "ref_path": "https://myrefs.com/sst-bert.json",
                    "data_descriptor": None,
                },
            ),
            self.normalize_ref({"ref_path": "https://myrefs.com/sst-bert.json"}),
        )

    def test_normalize_dict_with_data_id(self):
        self.assertEqual(
            (
                "sst-bibo",
                {
                    "ref_path": "https://myrefs.com/sst-cube.json",
                    "data_descriptor": None,
                },
            ),
            self.normalize_ref(
                {"ref_path": "https://myrefs.com/sst-cube.json", "data_id": "sst-bibo"}
            ),
        )

    def test_normalize_dict_with_data_descriptor(self):
        data_id, ref_dict = self.normalize_ref(
            {
                "ref_path": "https://myrefs.com/sst-cube.json",
                "data_descriptor": {"data_id": "sst-bibo", "data_type": "dataset"},
            }
        )
        self.assertEqual("sst-bibo", data_id)
        self.assertEqual("https://myrefs.com/sst-cube.json", ref_dict.get("ref_path"))
        self.assertIsInstance(ref_dict.get("data_descriptor"), DatasetDescriptor)
        self.assertEqual("sst-bibo", ref_dict.get("data_descriptor").data_id)

    def test_errors(self):
        with pytest.raises(TypeError, match="item in refs must be a str or a dict"):
            # noinspection PyTypeChecker
            self.normalize_ref(13)
        with pytest.raises(ValueError, match="missing key ref_path in refs item"):
            self.normalize_ref({})
        with pytest.raises(
            TypeError,
            match=(
                "value of data_descriptor key" " in refs item must be a dict or None"
            ),
        ):
            self.normalize_ref({"ref_path": "bibo", "data_descriptor": 13})


@unittest.skipUnless(has_kerchunk, reason="kerchunk not installed")
class ReferenceFilesystemTest(KerchunkMixin, unittest.TestCase):
    def test_xarray_open_zarr(self):
        cube_ref_path = self.ref_paths[0]
        fs = fsspec.filesystem("reference", fo=cube_ref_path, remote_protocol="file")
        self.assertEqual(
            {
                ".zattrs",
                ".zgroup",
                "time",
                "lat",
                "lon",
                "analysed_sst",
                "analysis_error",
                "mask",
                "sea_ice_fraction",
            },
            set(fs.ls("", detail=False)),
        )
        zarr_store = fs.get_mapper()
        cube = xr.open_zarr(zarr_store, consolidated=False)
        self.assert_sst_cube_ok(cube)

    def test_xarray_open_dataset(self):
        cube_ref_path = self.ref_paths[0]
        cube = xr.open_dataset(
            "reference://",
            engine="zarr",
            backend_kwargs=dict(
                consolidated=False,
                storage_options=dict(fo=cube_ref_path, remote_protocol="file"),
            ),
        )
        self.assert_sst_cube_ok(cube)
