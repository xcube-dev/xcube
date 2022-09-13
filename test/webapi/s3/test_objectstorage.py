# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
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

from xcube.core.mldataset import BaseMultiLevelDataset
from xcube.core.new import new_cube
# noinspection PyUnresolvedReferences
from xcube.core.zarrstore import ZarrStoreHolder
from xcube.webapi.s3.objectstorage import ObjectStorage


class ObjectStorageTest(unittest.TestCase):

    def setUp(self) -> None:
        self.maxDiff = None

        cube_1 = BaseMultiLevelDataset(
            new_cube(
                x_name='x',
                y_name='y',
                width=1000,
                height=1000,
                x_start=0,
                y_start=0,
                x_res=10,
                crs='EPSG:3035',
                variables={"B03": 0.2, "B04": 0.1, "B08": 0.1},
                drop_bounds=True,
            ).chunk(dict(x=250, y=250))
        )

        cube_2 = new_cube(
            width=3600,
            height=1800,
            x_res=0.1,
            variables={"CHL": 0.8, "TSM": 2.3},
            drop_bounds=True,
        ).chunk(dict(lon=900, lat=900))

        datasets = {
            "cube_1.levels": cube_1,
            "cube_2.zarr": cube_2
        }
        self.object_storage = ObjectStorage(datasets)

    def test_keys(self):
        self.assertEqual(
            EXPECTED_KEYS,
            sorted(self.object_storage.keys())
        )

    def test_iter(self):
        self.assertEqual(
            EXPECTED_KEYS,
            sorted(iter(self.object_storage))
        )

    def test_len(self):
        self.assertEqual(
            len(EXPECTED_KEYS),
            len(self.object_storage)
        )


EXPECTED_KEYS = [
    'cube_1.levels/0.zarr/.zattrs',
    'cube_1.levels/0.zarr/.zgroup',
    'cube_1.levels/0.zarr/.zmetadata',
    'cube_1.levels/0.zarr/B03/.zarray',
    'cube_1.levels/0.zarr/B03/.zattrs',
    'cube_1.levels/0.zarr/B03/0.0.0',
    'cube_1.levels/0.zarr/B03/0.0.1',
    'cube_1.levels/0.zarr/B03/0.0.2',
    'cube_1.levels/0.zarr/B03/0.0.3',
    'cube_1.levels/0.zarr/B03/0.1.0',
    'cube_1.levels/0.zarr/B03/0.1.1',
    'cube_1.levels/0.zarr/B03/0.1.2',
    'cube_1.levels/0.zarr/B03/0.1.3',
    'cube_1.levels/0.zarr/B03/0.2.0',
    'cube_1.levels/0.zarr/B03/0.2.1',
    'cube_1.levels/0.zarr/B03/0.2.2',
    'cube_1.levels/0.zarr/B03/0.2.3',
    'cube_1.levels/0.zarr/B03/0.3.0',
    'cube_1.levels/0.zarr/B03/0.3.1',
    'cube_1.levels/0.zarr/B03/0.3.2',
    'cube_1.levels/0.zarr/B03/0.3.3',
    'cube_1.levels/0.zarr/B04/.zarray',
    'cube_1.levels/0.zarr/B04/.zattrs',
    'cube_1.levels/0.zarr/B04/0.0.0',
    'cube_1.levels/0.zarr/B04/0.0.1',
    'cube_1.levels/0.zarr/B04/0.0.2',
    'cube_1.levels/0.zarr/B04/0.0.3',
    'cube_1.levels/0.zarr/B04/0.1.0',
    'cube_1.levels/0.zarr/B04/0.1.1',
    'cube_1.levels/0.zarr/B04/0.1.2',
    'cube_1.levels/0.zarr/B04/0.1.3',
    'cube_1.levels/0.zarr/B04/0.2.0',
    'cube_1.levels/0.zarr/B04/0.2.1',
    'cube_1.levels/0.zarr/B04/0.2.2',
    'cube_1.levels/0.zarr/B04/0.2.3',
    'cube_1.levels/0.zarr/B04/0.3.0',
    'cube_1.levels/0.zarr/B04/0.3.1',
    'cube_1.levels/0.zarr/B04/0.3.2',
    'cube_1.levels/0.zarr/B04/0.3.3',
    'cube_1.levels/0.zarr/B08/.zarray',
    'cube_1.levels/0.zarr/B08/.zattrs',
    'cube_1.levels/0.zarr/B08/0.0.0',
    'cube_1.levels/0.zarr/B08/0.0.1',
    'cube_1.levels/0.zarr/B08/0.0.2',
    'cube_1.levels/0.zarr/B08/0.0.3',
    'cube_1.levels/0.zarr/B08/0.1.0',
    'cube_1.levels/0.zarr/B08/0.1.1',
    'cube_1.levels/0.zarr/B08/0.1.2',
    'cube_1.levels/0.zarr/B08/0.1.3',
    'cube_1.levels/0.zarr/B08/0.2.0',
    'cube_1.levels/0.zarr/B08/0.2.1',
    'cube_1.levels/0.zarr/B08/0.2.2',
    'cube_1.levels/0.zarr/B08/0.2.3',
    'cube_1.levels/0.zarr/B08/0.3.0',
    'cube_1.levels/0.zarr/B08/0.3.1',
    'cube_1.levels/0.zarr/B08/0.3.2',
    'cube_1.levels/0.zarr/B08/0.3.3',
    'cube_1.levels/0.zarr/crs/.zarray',
    'cube_1.levels/0.zarr/crs/.zattrs',
    'cube_1.levels/0.zarr/crs/0',
    'cube_1.levels/0.zarr/time/.zarray',
    'cube_1.levels/0.zarr/time/.zattrs',
    'cube_1.levels/0.zarr/time/0',
    'cube_1.levels/0.zarr/x/.zarray',
    'cube_1.levels/0.zarr/x/.zattrs',
    'cube_1.levels/0.zarr/x/0',
    'cube_1.levels/0.zarr/y/.zarray',
    'cube_1.levels/0.zarr/y/.zattrs',
    'cube_1.levels/0.zarr/y/0',
    'cube_1.levels/1.zarr/.zattrs',
    'cube_1.levels/1.zarr/.zgroup',
    'cube_1.levels/1.zarr/.zmetadata',
    'cube_1.levels/1.zarr/B03/.zarray',
    'cube_1.levels/1.zarr/B03/.zattrs',
    'cube_1.levels/1.zarr/B03/0.0.0',
    'cube_1.levels/1.zarr/B03/0.0.1',
    'cube_1.levels/1.zarr/B03/0.1.0',
    'cube_1.levels/1.zarr/B03/0.1.1',
    'cube_1.levels/1.zarr/B04/.zarray',
    'cube_1.levels/1.zarr/B04/.zattrs',
    'cube_1.levels/1.zarr/B04/0.0.0',
    'cube_1.levels/1.zarr/B04/0.0.1',
    'cube_1.levels/1.zarr/B04/0.1.0',
    'cube_1.levels/1.zarr/B04/0.1.1',
    'cube_1.levels/1.zarr/B08/.zarray',
    'cube_1.levels/1.zarr/B08/.zattrs',
    'cube_1.levels/1.zarr/B08/0.0.0',
    'cube_1.levels/1.zarr/B08/0.0.1',
    'cube_1.levels/1.zarr/B08/0.1.0',
    'cube_1.levels/1.zarr/B08/0.1.1',
    'cube_1.levels/1.zarr/crs/.zarray',
    'cube_1.levels/1.zarr/crs/.zattrs',
    'cube_1.levels/1.zarr/crs/0',
    'cube_1.levels/1.zarr/time/.zarray',
    'cube_1.levels/1.zarr/time/.zattrs',
    'cube_1.levels/1.zarr/time/0',
    'cube_1.levels/1.zarr/x/.zarray',
    'cube_1.levels/1.zarr/x/.zattrs',
    'cube_1.levels/1.zarr/x/0',
    'cube_1.levels/1.zarr/y/.zarray',
    'cube_1.levels/1.zarr/y/.zattrs',
    'cube_1.levels/1.zarr/y/0',
    'cube_1.levels/2.zarr/.zattrs',
    'cube_1.levels/2.zarr/.zgroup',
    'cube_1.levels/2.zarr/.zmetadata',
    'cube_1.levels/2.zarr/B03/.zarray',
    'cube_1.levels/2.zarr/B03/.zattrs',
    'cube_1.levels/2.zarr/B03/0.0.0',
    'cube_1.levels/2.zarr/B04/.zarray',
    'cube_1.levels/2.zarr/B04/.zattrs',
    'cube_1.levels/2.zarr/B04/0.0.0',
    'cube_1.levels/2.zarr/B08/.zarray',
    'cube_1.levels/2.zarr/B08/.zattrs',
    'cube_1.levels/2.zarr/B08/0.0.0',
    'cube_1.levels/2.zarr/crs/.zarray',
    'cube_1.levels/2.zarr/crs/.zattrs',
    'cube_1.levels/2.zarr/crs/0',
    'cube_1.levels/2.zarr/time/.zarray',
    'cube_1.levels/2.zarr/time/.zattrs',
    'cube_1.levels/2.zarr/time/0',
    'cube_1.levels/2.zarr/x/.zarray',
    'cube_1.levels/2.zarr/x/.zattrs',
    'cube_1.levels/2.zarr/x/0',
    'cube_1.levels/2.zarr/y/.zarray',
    'cube_1.levels/2.zarr/y/.zattrs',
    'cube_1.levels/2.zarr/y/0',
    'cube_2.zarr/.zattrs',
    'cube_2.zarr/.zgroup',
    'cube_2.zarr/.zmetadata',
    'cube_2.zarr/CHL/.zarray',
    'cube_2.zarr/CHL/.zattrs',
    'cube_2.zarr/CHL/0.0.0',
    'cube_2.zarr/CHL/0.0.1',
    'cube_2.zarr/CHL/0.0.2',
    'cube_2.zarr/CHL/0.0.3',
    'cube_2.zarr/CHL/0.1.0',
    'cube_2.zarr/CHL/0.1.1',
    'cube_2.zarr/CHL/0.1.2',
    'cube_2.zarr/CHL/0.1.3',
    'cube_2.zarr/TSM/.zarray',
    'cube_2.zarr/TSM/.zattrs',
    'cube_2.zarr/TSM/0.0.0',
    'cube_2.zarr/TSM/0.0.1',
    'cube_2.zarr/TSM/0.0.2',
    'cube_2.zarr/TSM/0.0.3',
    'cube_2.zarr/TSM/0.1.0',
    'cube_2.zarr/TSM/0.1.1',
    'cube_2.zarr/TSM/0.1.2',
    'cube_2.zarr/TSM/0.1.3',
    'cube_2.zarr/lat/.zarray',
    'cube_2.zarr/lat/.zattrs',
    'cube_2.zarr/lat/0',
    'cube_2.zarr/lon/.zarray',
    'cube_2.zarr/lon/.zattrs',
    'cube_2.zarr/lon/0',
    'cube_2.zarr/time/.zarray',
    'cube_2.zarr/time/.zattrs',
    'cube_2.zarr/time/0'
]
