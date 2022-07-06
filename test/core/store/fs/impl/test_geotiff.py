# The MIT License (MIT)
# Copyright (c) 2021/2022 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os.path
import unittest

import fsspec
import rasterio as rio
import rioxarray
import s3fs
import xarray
import xarray as xr

from test.s3test import S3Test
from xcube.core.store.fs.impl.dataset import DatasetGeoTiffFsDataAccessor
from xcube.core.store.fs.impl.geotiff import GeoTIFFMultiLevelDataset
from xcube.core.store.fs.impl.geotiff import \
    MultiLevelDatasetGeoTiffFsDataAccessor
from xcube.util.jsonschema import JsonSchema


class RioXarrayTest(unittest.TestCase):
    """
    This class doesn't test xcube but rather asserts that RioXarray works as
    expected
    """

    def test_it_is_possible_get_overviews(self):
        cog_path = os.path.join(os.path.dirname(__file__),
                                "..", "..", "..", "..", "..",
                                "examples", "serve", "demo",
                                "sample-cog.tif")
        array = rioxarray.open_rasterio(cog_path)
        self.assertIsInstance(array, xr.DataArray)
        self.assertEqual((3, 343, 343), array.shape)

        rio_accessor = array.rio
        self.assertIsInstance(rio_accessor,
                              rioxarray.raster_array.RasterArray)
        manager = rio_accessor._manager
        print(manager)
        self.assertIsInstance(manager,
                              xr.backends.CachingFileManager)
        rio_dataset = manager.acquire(needs_lock=False)
        self.assertIsInstance(rio_dataset,
                              rio.DatasetReader)

        overviews = rio_dataset.overviews(1)
        self.assertEqual([2, 4], overviews)

        num_levels = len(overviews)
        shapes = []
        for i in range(num_levels):
            array = rioxarray.open_rasterio(cog_path,
                                            overview_level=i)
            shapes.append(array.shape)

        self.assertEqual([(3, 172, 172), (3, 86, 86)],
                         shapes)

        with self.assertRaises(rio.errors.RasterioIOError) as e:
            rioxarray.open_rasterio(cog_path,
                                    overview_level=num_levels)

        self.assertTrue(f'{e.exception}'.startswith(
            'Cannot open overview level 2 of '
        ))


class GeoTIFFMultiLevelDatasetTest(unittest.TestCase):
    """
    A class to test wrapping of geotiff file into a multilevel dataset
    """

    @classmethod
    def get_params(cls, file_name):
        fs = fsspec.filesystem('file')
        cog_path = os.path.join(os.path.dirname(__file__),
                                "..", "..", "..", "..", "..",
                                "examples", "serve", "demo",
                                file_name)
        return fs, cog_path

    def test_local_fs(self):
        fs, cog_path = self.get_params("sample-cog.tif")
        ml_dataset = GeoTIFFMultiLevelDataset(fs, None, cog_path)
        self.assertIsInstance(ml_dataset, GeoTIFFMultiLevelDataset)
        self.assertEqual(3, ml_dataset.num_levels)
        self.assertEqual(cog_path, ml_dataset.ds_id)
        self.assertEqual([(320, 320), (640, 640), (1280, 1280)],
                         ml_dataset.resolutions)
        dataset_1 = ml_dataset.get_dataset(0)
        self.assertEqual(['band_1', 'band_2', 'band_3'],
                         list(dataset_1.data_vars))
        self.assertEqual((343, 343),
                         dataset_1.band_1.shape)
        dataset_2 = ml_dataset.get_dataset(2)
        self.assertEqual(['band_1', 'band_2', 'band_3'],
                         list(dataset_2.data_vars))
        self.assertEqual((86, 86),
                         dataset_2.band_1.shape)
        datasets = ml_dataset.datasets
        self.assertEqual(3, len(datasets))


class MultiLevelDatasetGeoTiffFsDataAccessorTest(unittest.TestCase):
    """
    A class to test cog and GeoTIFF for multilevel dataset opener
    """

    def test_read_cog(self):
        fs, cog_path = GeoTIFFMultiLevelDatasetTest.get_params("sample-cog.tif")
        ml_data_opener = MultiLevelDatasetGeoTiffFsDataAccessor()
        ml_dataset = ml_data_opener.open_data(cog_path, fs=fs, root=None,
                                              tile_size=[256, 256])
        self.assertIsInstance(ml_dataset, GeoTIFFMultiLevelDataset)
        self.assertEqual(3, ml_dataset.num_levels)
        dataset = ml_dataset.get_dataset(0)
        self.assertEqual(['band_1', 'band_2', 'band_3'],
                         list(dataset.data_vars))
        self.assertEqual((343, 343), dataset.band_1.shape)
        self.assertEqual(2, len(dataset.dims))
        self.assertEqual("geotiff", ml_data_opener.get_format_id())
        self.assertIsInstance(
            ml_data_opener.get_open_data_params_schema(cog_path),
            JsonSchema
        )
        self.assertIsInstance(ml_data_opener.get_open_data_params_schema(),
                              JsonSchema)

    def test_read_geotiff(self):
        fs, tiff_path = GeoTIFFMultiLevelDatasetTest. \
            get_params("sample-geotiff.tif")
        data_opener = MultiLevelDatasetGeoTiffFsDataAccessor()

        ml_dataset = data_opener.open_data(tiff_path, fs=fs, root=None,
                                           tile_size=[512, 512])
        self.assertIsInstance(ml_dataset, GeoTIFFMultiLevelDataset)

        self.assertEqual(1, ml_dataset.num_levels)
        dataset = ml_dataset.get_dataset(0)
        self.assertEqual((1387, 1491), dataset.band_1.shape)


class DatasetGeoTiffFsDataAccessorTest(unittest.TestCase):
    """
    A Test class to test opening a GeoTIFF as multilevel dataset or 
    as normal dataset
    """

    def test_ml_to_dataset(self):
        fs, cog_path = GeoTIFFMultiLevelDatasetTest.get_params("sample-cog.tif")
        data_accessor = DatasetGeoTiffFsDataAccessor()
        self.assertIsInstance(data_accessor, DatasetGeoTiffFsDataAccessor)
        self.assertEqual("geotiff", data_accessor.get_format_id())
        dataset = data_accessor.open_data(data_id=cog_path, overview_level=1,
                                          fs=fs, root=None,
                                          tile_size=[512, 512])
        self.assertIsInstance(dataset, xarray.Dataset)
        self.assertIsInstance(
            data_accessor.get_open_data_params_schema(cog_path),
            JsonSchema
        )

    def test_dataset(self):
        fs, tiff_path = GeoTIFFMultiLevelDatasetTest. \
            get_params("sample-geotiff.tif")
        data_accessor = DatasetGeoTiffFsDataAccessor()
        self.assertIsInstance(data_accessor, DatasetGeoTiffFsDataAccessor)
        self.assertEqual("geotiff", data_accessor.get_format_id())
        dataset = data_accessor.open_data(data_id=tiff_path,
                                          fs=fs, root=None,
                                          tile_size=[256, 256],
                                          overview_level=None)
        self.assertIsInstance(dataset, xarray.Dataset)


class ObjectStorageMultiLevelDatasetTest(S3Test):
    """
    A Test class to test opening a cog file from AWS S3
    """

    def test_s3_fs(self):
        region_name = 'eu-central-1'
        s3 = s3fs.S3FileSystem(region_name=region_name, anon=True)
        data_id = "xcube-examples/sample-cog.tif"
        ml_dataset = GeoTIFFMultiLevelDataset(s3, None, data_id)
        self.assertEqual(3, ml_dataset.num_levels)
        self.assertEqual((343, 343), ml_dataset.get_dataset(0).band_1.shape)
        dataset_path = "xcube-examples/sample-geotiff.tif"
        data_accessor = DatasetGeoTiffFsDataAccessor()
        dataset = data_accessor.open_data(data_id=dataset_path,
                                          fs=s3, root=None,
                                          tile_size=[256, 256],
                                          overview_level=None)
        self.assertIsInstance(dataset, xarray.Dataset)