# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest
from test.sampledata import SourceDatasetMixin

import numpy as np
import pyproj
import xarray as xr

from xcube.core.gridmapping import CRS_WGS84, GridMapping
from xcube.core.resampling import reproject_dataset


class ReprojectDatasetTest(SourceDatasetMixin, unittest.TestCase):
    def test_reproject_value_error(self):
        source_ds = self.new_5x5_dataset_regular_utm()
        with self.assertRaises(ValueError) as context:
            reproject_dataset(source_ds)
        self.assertEqual(
            str(context.exception), "Either ref_ds or target_gm needs to be given."
        )

    def test_reproject_reference_dataset(self):
        source_ds = self.new_5x5_dataset_regular_utm()

        # test reproject to reference dataset
        x = np.arange(4320120.0, 4320120.0 + 4.1 * 80.0, 80.0)
        y = np.arange(3382520.0 + 4.0 * 80.0, 3382519.0, -80.0)
        spatial_ref = np.array(0)
        band_1 = np.arange(25).reshape((5, 5))
        ref_ds = xr.Dataset(
            dict(
                band_1=xr.DataArray(
                    band_1, dims=("y", "x"), attrs=dict(grid_mapping="spatial_ref")
                )
            ),
            coords=dict(x=x, y=y, spatial_ref=spatial_ref),
        )
        ref_ds.spatial_ref.attrs = pyproj.CRS.from_epsg("3035").to_cf()
        target_ds = reproject_dataset(source_ds, ref_ds=ref_ds)
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

    def test_reproject_target_gm(self):
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

    def test_reproject_target_gm_j_axis_up(self):
        source_ds = self.new_5x5_dataset_regular_utm()
        target_gm = GridMapping.regular(
            size=(5, 5),
            xy_min=(4320080, 3382480),
            xy_res=80,
            crs="epsg:3035",
            is_j_axis_up=True,
        )
        target_ds = reproject_dataset(source_ds, target_gm=target_gm)
        np.testing.assert_almost_equal(
            target_ds.band_1.values,
            np.array(
                [
                    [21, 17, 17, 18, 19],
                    [16, 17, 17, 18, 19],
                    [11, 12, 12, 13, 14],
                    [6, 6, 7, 8, 9],
                    [1, 1, 2, 3, 4],
                ],
                dtype=target_ds.band_1.dtype,
            ),
        )

    def test_reproject_target_gm_finer_res(self):
        source_ds = self.new_5x5_dataset_regular_utm()

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

    def test_reproject_target_gm_coarser_res(self):
        source_ds = self.new_5x5_dataset_regular_utm()

        # test projected CRS finer resolution
        # test if subset calculation works as expected
        target_gm = GridMapping.regular(
            size=(3, 3), xy_min=(4320050, 3382500), xy_res=120, crs="epsg:3035"
        )
        target_ds = reproject_dataset(source_ds, target_gm=target_gm)
        np.testing.assert_almost_equal(
            target_ds.band_1.values,
            np.array(
                [
                    [0, 1, 2],
                    [10, 11, 12],
                    [15, 16, 17],
                ],
                dtype=target_ds.band_1.dtype,
            ),
        )

    def test_reproject_target_gm_geographic_crs(self):
        source_ds = self.new_5x5_dataset_regular_utm()
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

    def test_reproject_target_gm_geographic_crs_fine_res(self):
        source_ds = self.new_5x5_dataset_regular_utm()

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

    def test_reproject_complex_dask_array(self):
        source_ds = self.new_complex_dataset_for_reproject()
        target_gm = GridMapping.regular(
            size=(10, 10),
            xy_min=(6.0, 48.0),
            xy_res=0.2,
            crs=CRS_WGS84,
            tile_size=(5, 5),
        )

        target_ds = reproject_dataset(
            source_ds, target_gm=target_gm, interpolation="triangular"
        )
        self.assertCountEqual(["temperature"], list(target_ds.data_vars))
        self.assertEqual(target_ds.temperature.values[0, 0, 0], 6427.7188)
        self.assertEqual(target_ds.temperature.values[0, -1, -1], 3085.9507)
        self.assertEqual(
            [2, 5, 5],
            [
                target_ds.temperature.chunksizes["time"][0],
                target_ds.temperature.chunksizes["lat"][0],
                target_ds.temperature.chunksizes["lon"][0],
            ],
        )

        target_ds = reproject_dataset(
            source_ds, target_gm=target_gm, interpolation="bilinear"
        )
        self.assertCountEqual(["temperature"], list(target_ds.data_vars))
        self.assertEqual(target_ds.temperature.values[0, 0, 0], 6427.718652710034)
        self.assertEqual(target_ds.temperature.values[0, -1, -1], 3085.9507290783004)
        self.assertEqual(
            [2, 5, 5],
            [
                target_ds.temperature.chunksizes["time"][0],
                target_ds.temperature.chunksizes["lat"][0],
                target_ds.temperature.chunksizes["lon"][0],
            ],
        )
