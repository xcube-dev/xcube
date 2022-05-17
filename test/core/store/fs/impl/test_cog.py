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

# from xcube.core.store.fs.impl import cog
# from test.s3test import S3Test, MOTO_SERVER_ENDPOINT_URL
from xcube.core.store.fs.impl.cog import GeoTIFFMultiLevelDataset, \
    MultiLevelDatasetGeoTiffFsDataAccessor
from xcube.core.store.fs.impl.dataset import DatasetGeoTiffFsDataAccessor
from xcube.util.jsonschema import JsonObjectSchema


class RioXarrayTest(unittest.TestCase):

    def test_it_is_possible_get_overviews(self):
        cog_path = 'examples/serve/demo/cog-example.tif'
        array = rioxarray.open_rasterio(cog_path)
        self.assertIsInstance(array, xr.DataArray)
        self.assertEqual((3, 9984, 22016), array.shape)

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
        self.assertEqual([2, 4, 8, 16, 32, 64],
                         overviews)

        num_levels = len(overviews)
        shapes = []
        for i in range(num_levels):
            array = rioxarray.open_rasterio(cog_path,
                                            overview_level=i)
            shapes.append(array.shape)

        self.assertEqual([(3, 4992, 11008),
                          (3, 2496, 5504),
                          (3, 1248, 2752),
                          (3, 624, 1376),
                          (3, 312, 688),
                          (3, 156, 344)],
                         shapes)

        with self.assertRaises(rio.errors.RasterioIOError) as e:
            rioxarray.open_rasterio(cog_path,
                                    overview_level=num_levels)
        self.assertEqual(
            (
                'Cannot open overview level 6 of '
                'examples/serve/demo/cog-example.tif',),
            e.exception.args
        )


class GeoTIFFMultiLevelDatasetTest(unittest.TestCase):

    def test_local_fs(self):
        fs = fsspec.filesystem('file')
        cog_path = "examples/serve/demo/cog-example.tif"
        ml_dataset = GeoTIFFMultiLevelDataset(fs, None, cog_path)
        self.assertEqual(7, ml_dataset.num_levels)
        self.assertEqual(cog_path, ml_dataset.ds_id)
        self.assertEqual([(0.03732275, 0.03732275),
                          (0.0746455, 0.0746455),
                          (0.149291, 0.149291),
                          (0.298582, 0.298582),
                          (0.597164, 0.597164),
                          (1.194328, 1.194328),
                          (2.388656, 2.388656)], ml_dataset.resolutions)
        dataset = ml_dataset.get_dataset(0)
        self.assertEqual(['band_1', 'band_2', 'band_3'],
                         list(dataset.data_vars))
        self.assertEqual((9984, 22016),
                         dataset.band_1.shape)
        dataset = ml_dataset.get_dataset(6)
        self.assertEqual(['band_1', 'band_2', 'band_3'],
                         list(dataset.data_vars))
        self.assertEqual((9984 // 2 ** 6, 22016 // 2 ** 6),
                         dataset.band_1.shape)
        datasets = ml_dataset.datasets
        self.assertEqual(7, len(datasets))


# class ObjectStorageMultiLevelDatasetTest(S3Test):
#     def test_s3_fs(self):
#         s3 = s3fs.S3FileSystem(client_kwargs=dict(
#             endpoint_url=MOTO_SERVER_ENDPOINT_URL))
#
#         cog_path = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/" \
#                    "sentinel-s2-l2a-cogs/13/S/DV/2020/4/" \
#                    "S2B_13SDV_20200428_0_L2A/L2A_PVI.tif"
#         ml_dataset = GeoTIFFMultiLevelDataset(s3, None, cog_path)
#         self.assertEqual(3, ml_dataset.num_levels)
#
#     def test_s3_fs_1(self):
#         cog_filename = "cog-example.tif"
#         local_cog_path = os.path.join(os.path.dirname(__file__),
#                                       "..", "..", "..", "..", "..",
#                                       "examples", "serve", "demo",
#                                       cog_filename)
#         # Assert that local_cog_path is valid
#         self.assertEqual(True, os.path.exists(local_cog_path))
#
#         remote_bucket = 'xcube-cog-test'
#         remote_cog_path = f'{remote_bucket}/{cog_filename}'
#
#         s3 = s3fs.S3FileSystem(client_kwargs=dict(
#             endpoint_url=MOTO_SERVER_ENDPOINT_URL)
#         )
#         # create test bucket
#         s3.mkdir(remote_bucket)
#         s3.put_file(local_cog_path, remote_cog_path)
#         # Assert that it is now in S3
#         self.assertEqual(True, s3.isfile(remote_cog_path))
#
#         with open(local_cog_path, mode="rb") as fp:
#             local_bytes = fp.read()
#         with s3.open(remote_cog_path, mode="rb") as fp:
#             remote_bytes = fp.read()
#         # Assert local and remote bytes are equal
#         self.assertEqual(local_bytes, remote_bytes)
#
#         # Assert we can use GeoTIFFMultiLevelDataset to open it
#         ml_dataset = GeoTIFFMultiLevelDataset(s3,
#                                               remote_bucket,
#                                               cog_filename)
#         self.assertEqual(7, ml_dataset.num_levels)


class MultiLevelDatasetGeoTiffFsDataAccessorTest(unittest.TestCase):

    def test_read_cog(self):
        fs = fsspec.filesystem('file')
        mldataopener = MultiLevelDatasetGeoTiffFsDataAccessor()
        cog_path = 'examples/serve/demo/cog-example.tif'
        ml_dataset = mldataopener.open_data(cog_path, fs=fs, root=None)
        self.assertEqual(7, ml_dataset.num_levels)
        dataset = ml_dataset.get_dataset(0)
        self.assertEqual(['band_1', 'band_2', 'band_3'],
                         list(dataset.data_vars))
        self.assertEqual(len(dataset.dims), 2)
        self.assertEqual(mldataopener.get_format_id(), "geotiff")
        self.assertEqual(
            type(mldataopener.get_open_data_params_schema(cog_path)),
            JsonObjectSchema)


class DatasetGeoTiffFsDataAccessorTest(unittest.TestCase):

    def test_it(self):
        fs = fsspec.filesystem('file')
        cog_path = 'examples/serve/demo/cog-example.tif'
        data_accessor = DatasetGeoTiffFsDataAccessor()
        self.assertEqual(data_accessor.get_format_id(), "geotiff")
        dataset = data_accessor.open_data(data_id=cog_path, fs=fs, root=None)
        self.assertEqual(type(dataset), xarray.DataArray)
        self.assertEqual(
            type(data_accessor.get_open_data_params_schema(cog_path)),
            JsonObjectSchema)
