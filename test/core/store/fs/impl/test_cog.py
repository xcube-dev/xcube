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

import unittest

import fsspec
import rasterio as rio
import rioxarray
import xarray as xr

from xcube.core.store.fs.impl.cog import GeoTIFFMultiLevelDataset


class RioXarrayTest(unittest.TestCase):

    def test_it_is_possible_get_overviews(self):
        cog_path = 'examples/serve/demo/cog-example.tif'
        # array = rioxarray.open_rasterio('examples/serve/demo/cog-example.tif')
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
