# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest
from test.sampledata import SourceDatasetMixin

import numpy as np

from xcube.core.gridmapping import CRS_WGS84, GridMapping
from xcube.core.resampling import reproject_dataset


class ReprojectDatasetTest(SourceDatasetMixin, unittest.TestCase):
    def test_reproject_utm(self):
        source_ds = self.new_5x5_dataset_regular_utm()

        # test projected CRS similar resolution
        target_gm = GridMapping.regular(
            size=(5, 5), xy_min=(4320080, 3382480), xy_res=80, crs="epsg:3035"
        )
        target_ds = reproject_dataset(source_ds, target_gm=target_gm)
        np.testing.assert_almost_equal(
            target_ds.band_1.values,
            np.array(
                [
                    [1, 1, 2, 3, 4],
                    [6, 6, 7, 8, 9],
                    [11, 12, 12, 13, 14],
                    [16, 17, 17, 18, 19],
                    [21, 17, 17, 18, 19],
                ],
                dtype=target_ds.band_1.dtype,
            ),
        )

        # test projected CRS finer resolution
        # test if subset calculation works as expected
        target_gm = GridMapping.regular(
            size=(5, 5), xy_min=(4320080, 3382480), xy_res=20, crs="epsg:3035"
        )
        target_ds = reproject_dataset(source_ds, target_gm=target_gm)
        np.testing.assert_almost_equal(
            target_ds.band_1.values,
            np.array(
                [
                    [15, 16, 16, 16, 16],
                    [15, 16, 16, 16, 16],
                    [15, 16, 16, 16, 16],
                    [20, 21, 21, 21, 21],
                    [20, 21, 21, 21, 21],
                ],
                dtype=target_ds.band_1.dtype,
            ),
        )

        # test geographic CRS with similar resolution
        target_gm = GridMapping.regular(
            size=(5, 5), xy_min=(9.9886, 53.5499), xy_res=0.0006, crs=CRS_WGS84
        )
        target_ds = reproject_dataset(source_ds, target_gm=target_gm)
        np.testing.assert_almost_equal(
            target_ds.band_1.values,
            np.array(
                [
                    [7, 8, 8, 8, 9],
                    [12, 13, 13, 13, 14],
                    [12, 13, 13, 13, 14],
                    [17, 18, 18, 18, 19],
                    [22, 23, 23, 23, 24],
                ],
                dtype=target_ds.band_1.dtype,
            ),
        )

        # test geographic CRS with 1/2 resolution
        # test if subset calculation works as expected
        target_gm = GridMapping.regular(
            size=(5, 5), xy_min=(9.9886, 53.5499), xy_res=0.0003, crs=CRS_WGS84
        )
        target_ds = reproject_dataset(source_ds, target_gm=target_gm)
        np.testing.assert_almost_equal(
            target_ds.band_1.values,
            np.array(
                [
                    [12, 12, 12, 13, 13],
                    [17, 17, 17, 18, 18],
                    [17, 17, 17, 18, 18],
                    [22, 17, 17, 18, 18],
                    [22, 22, 22, 23, 23],
                ],
                dtype=target_ds.band_1.dtype,
            ),
        )
