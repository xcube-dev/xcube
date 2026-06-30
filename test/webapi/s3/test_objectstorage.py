# Copyright (c) 2018-2026 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

from xcube.core.mldataset import BaseMultiLevelDataset
from xcube.core.new import new_cube

# noinspection PyUnresolvedReferences
from xcube.webapi.s3.objectstorage import ObjectStorage


class ObjectStorageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.maxDiff = None

        cube_1 = BaseMultiLevelDataset(
            new_cube(
                x_name="x",
                y_name="y",
                width=1000,
                height=1000,
                x_start=0,
                y_start=0,
                x_res=10,
                crs="EPSG:3035",
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

        datasets = {"cube_1.levels": cube_1, "cube_2.zarr": cube_2}
        self.object_storage = ObjectStorage(datasets)

    def test_keys(self):
        self.assertEqual(EXPECTED_KEYS, sorted(self.object_storage.keys()))

    def test_iter(self):
        self.assertEqual(EXPECTED_KEYS, sorted(iter(self.object_storage)))

    def test_len(self):
        self.assertEqual(len(EXPECTED_KEYS), len(self.object_storage))


EXPECTED_KEYS = [
    "cube_1.levels/0.zarr/B03",
    "cube_1.levels/0.zarr/B04",
    "cube_1.levels/0.zarr/B08",
    "cube_1.levels/0.zarr/crs",
    "cube_1.levels/1.zarr/B03",
    "cube_1.levels/1.zarr/B04",
    "cube_1.levels/1.zarr/B08",
    "cube_1.levels/1.zarr/crs",
    "cube_1.levels/2.zarr/B03",
    "cube_1.levels/2.zarr/B04",
    "cube_1.levels/2.zarr/B08",
    "cube_1.levels/2.zarr/crs",
    "cube_2.zarr/CHL",
    "cube_2.zarr/TSM",
]
