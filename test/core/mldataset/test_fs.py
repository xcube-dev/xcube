# The MIT License (MIT)
# Copyright (c) 2023 by the xcube team and contributors
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


import math
import unittest
from typing import Optional, List, Mapping

import fsspec
import fsspec.core
import xarray as xr

from xcube.core.mldataset import FsMultiLevelDataset
from xcube.core.new import new_cube
from xcube.core.subsampling import AggMethod


class FsMultiLevelDatasetTest(unittest.TestCase):

    def setUp(self) -> None:
        self.dataset = new_cube(
            width=512,
            height=256,
            x_res=360 / 512,
            y_res=180 / 256,
            time_periods=1,
            variables=dict(CHL=0.8, qflags=1)
        )

        self.fs: fsspec.AbstractFileSystem = fsspec.core.get_filesystem_class(
            "memory"
        )()

    def tearDown(self) -> None:
        self.fs.rm("/", recursive=True)

    def test_io_nl_4_ts_256(self):
        self.assert_io_ok(
            "test.levels",
            num_levels=4,
            agg_methods=None,
            tile_size=256,
            use_saved_levels=False,
            base_dataset_path=None,
            expected_files=[
                ".zlevels",
                "0.zarr",
                "1.zarr",
                "2.zarr",
                "3.zarr"
            ],
            expected_num_levels=4,
            expected_tile_size=[256, 256],
            expected_agg_methods={'CHL': 'mean', 'qflags': 'first'},
        )

    def test_io_nl_4_ts_256_agg(self):
        self.assert_io_ok(
            "test.levels",
            num_levels=4,
            agg_methods={'CHL': 'median', 'qflags': 'max'},
            tile_size=256,
            use_saved_levels=False,
            base_dataset_path=None,
            expected_files=[
                ".zlevels",
                "0.zarr",
                "1.zarr",
                "2.zarr",
                "3.zarr"
            ],
            expected_num_levels=4,
            expected_tile_size=[256, 256],
            expected_agg_methods={'CHL': 'median', 'qflags': 'max'},
        )

    def test_io_nl_4_ts_256_base(self):
        self.dataset.to_zarr(self.fs.get_mapper("test.zarr"))
        self.assert_io_ok(
            "test.levels",
            num_levels=4,
            agg_methods=None,
            tile_size=256,
            use_saved_levels=False,
            base_dataset_path="test.zarr",
            expected_files=[
                ".zlevels",
                "0.link",
                "1.zarr",
                "2.zarr",
                "3.zarr"
            ],
            expected_num_levels=4,
            expected_tile_size=[256, 256],
            expected_agg_methods={'CHL': 'mean', 'qflags': 'first'},
        )

    def assert_io_ok(
            self,
            path: str,
            num_levels: Optional[int],
            agg_methods: Optional[Mapping[str, AggMethod]],
            tile_size: Optional[int],
            use_saved_levels: bool,
            base_dataset_path: Optional[str],
            expected_files: List[str],
            expected_num_levels: int,
            expected_agg_methods: Optional[Mapping[str, AggMethod]],
            expected_tile_size: List[int]
    ):
        fs = self.fs
        FsMultiLevelDataset.write_dataset(self.dataset,
                                          path,
                                          fs=fs,
                                          fs_root="",
                                          replace=True,
                                          num_levels=num_levels,
                                          agg_methods=agg_methods,
                                          tile_size=tile_size,
                                          use_saved_levels=use_saved_levels,
                                          base_dataset_path=base_dataset_path)
        self.assertTrue(fs.isdir(path))
        self.assertEqual(set([f"/{path}/{f}" for f in expected_files]),
                         set(fs.listdir(path, detail=False)))

        ml_dataset = FsMultiLevelDataset(path, fs=fs)
        self.assertEquals(expected_num_levels, ml_dataset.num_levels)
        self.assertEquals(expected_agg_methods, ml_dataset.agg_methods)
        self.assertEquals(expected_tile_size, ml_dataset.tile_size)
        self.assertEquals(use_saved_levels, ml_dataset.use_saved_levels)
        self.assertEquals(base_dataset_path, ml_dataset.base_dataset_path)
        self.assertEquals(None, ml_dataset.cache_size)
        self.assertEquals(num_levels, len(ml_dataset.size_weights))
        self.assertTrue(all(ml_dataset.size_weights > 0.0))
        for i in range(num_levels):
            self.assertIsInstance(ml_dataset.get_dataset(i), xr.Dataset)

    def test_compute_size_weights(self):
        size = 2 ** 28
        weighted_sizes = list(map(
            math.ceil, size * FsMultiLevelDataset.compute_size_weights(5)
        ))
        self.assertEqual(
            [201523393,
             50380849,
             12595213,
             3148804,
             787201],
            weighted_sizes
        )
