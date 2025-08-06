# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import os.path
import unittest
from test.s3test import S3Test

import dask
import fsspec
import rasterio as rio
import rioxarray
import s3fs
import xarray
import xarray as xr

from xcube.core.store.fs.impl.rasterio import (DatasetGeoTiffFsDataAccessor,
                                               DatasetJ2kFsDataAccessor,
                                               MultiLevelDatasetGeoTiffFsDataAccessor,
                                               MultiLevelDatasetJ2kFsDataAccessor,
                                               RasterioMultiLevelDataset)
from xcube.util.jsonschema import JsonSchema

_COG_TEST_FILE = "sample-cog.tif"
_JPEG2000_TEST_FILE = "sample.jp2"
_JPEG2000_SINGLE_BAND_TEST_FILE = "sample-sb.jp2"
_GEOTIFF_TEST_FILE = "sample-geotiff.tif"


class RioXarrayTest(unittest.TestCase):
    """
    This class doesn't test xcube but rather asserts that RioXarray works as
    expected
    """

    def test_it_is_possible_get_overviews(self):
        cog_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "..",
            "examples",
            "serve",
            "demo",
            _COG_TEST_FILE,
        )
        array = rioxarray.open_rasterio(cog_path)
        self.assertIsInstance(array, xr.DataArray)
        self.assertEqual((3, 343, 343), array.shape)

        rio_accessor = array.rio
        self.assertIsInstance(rio_accessor, rioxarray.raster_array.RasterArray)
        manager = rio_accessor._manager
        self.assertIsInstance(manager, xr.backends.CachingFileManager)
        rio_dataset = manager.acquire(needs_lock=False)
        self.assertIsInstance(rio_dataset, rio.DatasetReader)

        overviews = rio_dataset.overviews(1)
        self.assertEqual([2, 4], overviews)

        num_levels = len(overviews)
        shapes = []
        for i in range(num_levels):
            array = rioxarray.open_rasterio(cog_path, overview_level=i)
            shapes.append(array.shape)

        self.assertEqual([(3, 172, 172), (3, 86, 86)], shapes)

        with self.assertRaises(rio.errors.RasterioIOError) as e:
            rioxarray.open_rasterio(cog_path, overview_level=num_levels)

        self.assertTrue(f"{e.exception}".startswith("Cannot open overview level 2 of "))


class RasterIoMultiLevelDatasetTest(unittest.TestCase):
    """
    A class to test wrapping of a file into a multilevel dataset
    """

    @classmethod
    def get_params(cls, file_name, attach_filename=True):
        fs = fsspec.filesystem("file")
        file_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "..",
            "examples",
            "serve",
            "demo"
        )
        if attach_filename:
            file_path = os.path.join(file_path, file_name)
        return fs, file_path

    def test_local_fs(self):
        fs, cog_path = self.get_params(_COG_TEST_FILE)
        ml_dataset = RasterioMultiLevelDataset(fs, None, cog_path)
        self.assertIsInstance(ml_dataset, RasterioMultiLevelDataset)
        self.assertEqual(3, ml_dataset.num_levels)
        self.assertEqual(cog_path, ml_dataset.ds_id)
        self.assertEqual([(320, 320), (640, 640), (1280, 1280)], ml_dataset.resolutions)
        dataset_1 = ml_dataset.get_dataset(0)
        self.assertEqual(["band_1", "band_2", "band_3"], list(dataset_1.data_vars))
        self.assertEqual((343, 343), dataset_1.band_1.shape)
        dataset_2 = ml_dataset.get_dataset(2)
        self.assertEqual(["band_1", "band_2", "band_3"], list(dataset_2.data_vars))
        self.assertEqual((86, 86), dataset_2.band_1.shape)
        datasets = ml_dataset.datasets
        self.assertEqual(3, len(datasets))


class MultiLevelDatasetGeoTiffFsDataAccessorTest(unittest.TestCase):
    """
    A class to test cog and GeoTIFF for multilevel dataset opener
    """

    def test_read_cog(self):
        fs, cog_path = RasterIoMultiLevelDatasetTest.get_params(_COG_TEST_FILE)
        ml_data_opener = MultiLevelDatasetGeoTiffFsDataAccessor()
        ml_dataset = ml_data_opener.open_data(
            cog_path, fs=fs, root=None, tile_size=[256, 256]
        )
        self.assertIsInstance(ml_dataset, RasterioMultiLevelDataset)
        self.assertEqual(3, ml_dataset.num_levels)
        dataset = ml_dataset.get_dataset(0)
        self.assertEqual(["band_1", "band_2", "band_3"], list(dataset.data_vars))
        self.assertEqual((343, 343), dataset.band_1.shape)
        self.assertEqual(2, len(dataset.sizes))
        self.assertEqual("geotiff", ml_data_opener.get_format_id())
        self.assertIsInstance(
            ml_data_opener.get_open_data_params_schema(cog_path), JsonSchema
        )
        self.assertIsInstance(ml_data_opener.get_open_data_params_schema(), JsonSchema)

    def test_read_geotiff(self):
        fs, tiff_path = RasterIoMultiLevelDatasetTest.get_params(_GEOTIFF_TEST_FILE)
        data_opener = MultiLevelDatasetGeoTiffFsDataAccessor()

        ml_dataset = data_opener.open_data(
            tiff_path, fs=fs, root=None, tile_size=[512, 512]
        )
        self.assertIsInstance(ml_dataset, RasterioMultiLevelDataset)

        self.assertEqual(1, ml_dataset.num_levels)
        dataset = ml_dataset.get_dataset(0)
        self.assertEqual((1387, 1491), dataset.band_1.shape)


class MultiLevelDatasetJ2kFsDataAccessorTest(unittest.TestCase):
    """
    A class to test JPEG 2000 for multilevel dataset opener
    """

    def setUp(self) -> None:
        region_name = "eu-central-1"
        self._s3 = s3fs.S3FileSystem(region_name=region_name, anon=True)
        self._dask_scheduler = dask.config.get("scheduler", None)
        dask.config.set(scheduler="single-threaded")

    def tearDown(self):
        dask.config.set(scheduler=self._dask_scheduler)

    def test_read(self):
        fs, file_path = RasterIoMultiLevelDatasetTest.get_params(_JPEG2000_TEST_FILE)
        data_opener = MultiLevelDatasetJ2kFsDataAccessor()

        ml_dataset = data_opener.open_data(
            file_path, fs=fs, root=None, tile_size=[512, 512]
        )
        self.assertIsInstance(ml_dataset, RasterioMultiLevelDataset)

        self.assertEqual(3, ml_dataset.num_levels)
        dataset = ml_dataset.get_dataset(0)
        self.assertEqual((1387, 1491), dataset.band_1.shape)

    def test_read_single_band(self):
        fs, file_path = RasterIoMultiLevelDatasetTest.get_params(
            _JPEG2000_SINGLE_BAND_TEST_FILE, False
        )
        data_opener = MultiLevelDatasetJ2kFsDataAccessor()

        ml_dataset = data_opener.open_data(
            _JPEG2000_SINGLE_BAND_TEST_FILE, fs=fs, root=file_path, tile_size=[512, 512]
        )
        self.assertIsInstance(ml_dataset, RasterioMultiLevelDataset)

        self.assertEqual(3, ml_dataset.num_levels)
        dataset = ml_dataset.get_dataset(0)
        self.assertEqual((1387, 1491), dataset.band_1.shape)


class DatasetGeoTiffFsDataAccessorTest(unittest.TestCase):
    """
    A Test class to test opening a GeoTIFF as multilevel dataset or
    as normal dataset
    """

    def test_ml_to_dataset(self):
        fs, cog_path = RasterIoMultiLevelDatasetTest.get_params(_COG_TEST_FILE)
        data_accessor = DatasetGeoTiffFsDataAccessor()
        self.assertIsInstance(data_accessor, DatasetGeoTiffFsDataAccessor)
        self.assertEqual("geotiff", data_accessor.get_format_id())
        dataset = data_accessor.open_data(
            data_id=cog_path, overview_level=1, fs=fs, root=None, tile_size=[512, 512]
        )
        self.assertIsInstance(dataset, xarray.Dataset)
        self.assertIsInstance(
            data_accessor.get_open_data_params_schema(cog_path), JsonSchema
        )

    def test_dataset(self):
        fs, tiff_path = RasterIoMultiLevelDatasetTest.get_params(_GEOTIFF_TEST_FILE)
        data_accessor = DatasetGeoTiffFsDataAccessor()
        self.assertIsInstance(data_accessor, DatasetGeoTiffFsDataAccessor)
        self.assertEqual("geotiff", data_accessor.get_format_id())
        dataset = data_accessor.open_data(
            data_id=tiff_path,
            fs=fs,
            root=None,
            tile_size=[256, 256],
            overview_level=None,
        )
        self.assertIsInstance(dataset, xarray.Dataset)

class DatasetJ2kFsDataAccessorTest(unittest.TestCase):
    """
    A Test class to test opening a JPEG 2000 as multilevel dataset or
    as normal dataset
    """

    def setUp(self) -> None:
        region_name = "eu-central-1"
        self._s3 = s3fs.S3FileSystem(region_name=region_name, anon=True)
        self._dask_scheduler = dask.config.get("scheduler", None)
        dask.config.set(scheduler="single-threaded")

    def tearDown(self):
        dask.config.set(scheduler=self._dask_scheduler)

    def test_ml_to_dataset(self):
        fs, file_path = RasterIoMultiLevelDatasetTest.get_params(_JPEG2000_TEST_FILE)
        data_accessor = DatasetJ2kFsDataAccessor()
        self.assertIsInstance(data_accessor, DatasetJ2kFsDataAccessor)
        self.assertEqual("j2k", data_accessor.get_format_id())
        dataset = data_accessor.open_data(
            data_id=file_path, overview_level=1, fs=fs, root=None, tile_size=[512, 512]
        )
        self.assertIsInstance(dataset, xarray.Dataset)
        self.assertIsInstance(
            data_accessor.get_open_data_params_schema(file_path), JsonSchema
        )

    def test_dataset(self):
        fs, file_path = RasterIoMultiLevelDatasetTest.get_params(_JPEG2000_TEST_FILE)
        data_accessor = DatasetJ2kFsDataAccessor()
        self.assertIsInstance(data_accessor, DatasetJ2kFsDataAccessor)
        self.assertEqual("j2k", data_accessor.get_format_id())
        dataset = data_accessor.open_data(
            data_id=file_path,
            fs=fs,
            root=None,
            tile_size=[256, 256],
            overview_level=None,
        )
        self.assertIsInstance(dataset, xarray.Dataset)

    def test_dataset_single_band(self):
        fs, file_path = RasterIoMultiLevelDatasetTest.get_params(
            _JPEG2000_SINGLE_BAND_TEST_FILE, False
        )
        data_accessor = DatasetJ2kFsDataAccessor()
        self.assertIsInstance(data_accessor, DatasetJ2kFsDataAccessor)
        self.assertEqual("j2k", data_accessor.get_format_id())
        dataset = data_accessor.open_data(
            data_id=_JPEG2000_SINGLE_BAND_TEST_FILE,
            fs=fs,
            root=file_path,
            tile_size=[256, 256],
            overview_level=None,
        )
        self.assertIsInstance(dataset, xarray.Dataset)


class ObjectStorageMultiLevelDatasetTest(S3Test):
    """
    A Test class to test opening files with rasterio from AWS S3
    """

    def setUp(self) -> None:
        region_name = "eu-central-1"
        self._s3 = s3fs.S3FileSystem(region_name=region_name, anon=True)
        self._dask_scheduler = dask.config.get("scheduler", None)
        dask.config.set(scheduler="single-threaded")

    def tearDown(self):
        dask.config.set(scheduler=self._dask_scheduler)

    def test_s3_fs_tif(self):
        data_id = f"xcube-examples/{_COG_TEST_FILE}"
        ml_dataset = RasterioMultiLevelDataset(self._s3, None, data_id)
        self.assertEqual(3, ml_dataset.num_levels)
        self.assertEqual((343, 343), ml_dataset.get_dataset(0).band_1.shape)
        dataset_path = f"xcube-examples/{_GEOTIFF_TEST_FILE}"
        data_accessor = DatasetGeoTiffFsDataAccessor()
        dataset = data_accessor.open_data(
            data_id=dataset_path,
            fs=self._s3,
            root=None,
            tile_size=[256, 256],
            overview_level=None,
        )
        self.assertIsInstance(dataset, xarray.Dataset)

    def test_s3_fs_jp2(self):
        data_id = f"xcube-examples/{_JPEG2000_TEST_FILE}"
        ml_dataset = RasterioMultiLevelDataset(self._s3, None, data_id)
        self.assertEqual(3, ml_dataset.num_levels)
        self.assertEqual((1387, 1491), ml_dataset.get_dataset(0).band_1.shape)
        dataset_path = f"xcube-examples/{_JPEG2000_TEST_FILE}"
        data_accessor = DatasetJ2kFsDataAccessor()
        dataset = data_accessor.open_data(
            data_id=dataset_path,
            fs=self._s3,
            root=None,
            tile_size=[256, 256],
            overview_level=None,
        )
        self.assertIsInstance(dataset, xarray.Dataset)
