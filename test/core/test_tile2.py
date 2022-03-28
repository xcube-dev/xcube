import unittest

import numpy as np
import pyproj
import xarray as xr

from xcube.core.mldataset import BaseMultiLevelDataset
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.tile2 import compute_rgba_tile
from xcube.core.tilegrid import GEOGRAPHIC_CRS_NAME
from xcube.core.tilegrid import WEB_MERCATOR_CRS_NAME


class Tile2Test(unittest.TestCase):

    @staticmethod
    def _get_ml_dataset(crs_name: str) -> MultiLevelDataset:
        crs = pyproj.CRS.from_string(crs_name)
        geo_crs = pyproj.CRS.from_string(GEOGRAPHIC_CRS_NAME)
        transformer = pyproj.Transformer.from_crs(
            geo_crs, crs, always_xy=True
        )
        (x1, x2), (y1, y2) = transformer.transform(
            (-180, 180),
            (-85.051129, 85.051129)
        )

        data = np.array(
            [[
                [5, 1, 1, 1, 1, 2, 2, 2, 2, 6],
                [1, 5, 1, 1, 1, 2, 2, 2, 6, 2],
                [1, 1, 5, 1, 1, 2, 2, 6, 2, 2],
                [1, 1, 1, 5, 1, 2, 6, 2, 2, 2],
                [1, 1, 1, 1, 5, 6, 2, 2, 2, 2],
                [3, 3, 3, 3, 6, 5, 4, 4, 4, 4],
                [3, 3, 3, 6, 3, 4, 5, 4, 4, 4],
                [3, 3, 6, 3, 3, 4, 4, 5, 4, 4],
                [3, 6, 3, 3, 3, 4, 4, 4, 5, 4],
                [6, 3, 3, 3, 3, 4, 4, 4, 4, 5],
            ]],
            dtype=np.float32
        )
        a_data = np.where(data != 6, data + 0, np.nan)
        b_data = np.where(data != 5, data + 1, np.nan)
        c_data = np.where(data != 4, data + 2, np.nan)
        h, w = a_data.shape[-2:]
        res = (x2 - x1) / w
        x1 += 0.5 * res
        x2 -= 0.5 * res
        y1 += 0.5 * res
        y2 -= 0.5 * res

        var_a = xr.DataArray(a_data, dims=['time', 'y', 'x'])
        var_b = xr.DataArray(b_data, dims=['time', 'y', 'x'])
        var_c = xr.DataArray(c_data, dims=['time', 'y', 'x'])
        spatial_ref = xr.DataArray(
            0, attrs=pyproj.CRS.from_string(crs_name).to_cf()
        )

        ds = xr.Dataset(
            data_vars=dict(
                var_a=var_a,
                var_b=var_b,
                var_c=var_c,
                spatial_ref=spatial_ref
            ),
            coords=dict(
                time=xr.DataArray(np.array([0]), dims='time'),
                y=xr.DataArray(np.linspace(y1, y2, h), dims='y'),
                x=xr.DataArray(np.linspace(x1, x2, w), dims='x')
            )
        )
        # ds = ds.chunk(dict(x=4, y=3))
        return BaseMultiLevelDataset(ds)

    def test_compute_rgba_tile_with_color_mapping(self):
        crs_name = WEB_MERCATOR_CRS_NAME
        ml_ds = self._get_ml_dataset(crs_name)
        args = [
            ml_ds,
            'var_a',
            0, 0, 0
        ]
        kwargs = dict(
            crs_name=crs_name,
            tile_size=10,
            cmap_name="gray",
            value_ranges=(0, 10),
            non_spatial_labels={'time': 0},
            tile_enlargement=0
        )
        tile = compute_rgba_tile(*args, **kwargs, format='numpy')
        self.assertIsInstance(tile, np.ndarray)
        self.assertEqual(np.uint8, tile.dtype)
        self.assertEqual((10, 10, 4), tile.shape)
        r = tile[:, :, 0]
        g = tile[:, :, 1]
        b = tile[:, :, 2]
        a = tile[:, :, 3]
        np.testing.assert_equal(
            r,
            np.array(
                [[0, 0, 76, 76, 76, 76, 102, 102, 102, 128],
                 [76, 76, 0, 76, 76, 76, 102, 102, 128, 102],
                 [76, 76, 76, 0, 76, 76, 102, 128, 102, 102],
                 [76, 76, 76, 76, 0, 0, 128, 102, 102, 102],
                 [25, 25, 25, 25, 128, 128, 0, 51, 51, 51],
                 [25, 25, 25, 25, 128, 128, 0, 51, 51, 51],
                 [25, 25, 25, 128, 25, 25, 51, 0, 51, 51],
                 [25, 25, 128, 25, 25, 25, 51, 51, 0, 51],
                 [128, 128, 25, 25, 25, 25, 51, 51, 51, 0],
                 [128, 128, 25, 25, 25, 25, 51, 51, 51, 0]],
                dtype=np.uint8
            )
        )
        np.testing.assert_equal(r, g)
        np.testing.assert_equal(r, b)
        np.testing.assert_equal(
            a,
            np.array(
                [[0, 0, 255, 255, 255, 255, 255, 255, 255, 255],
                 [255, 255, 0, 255, 255, 255, 255, 255, 255, 255],
                 [255, 255, 255, 0, 255, 255, 255, 255, 255, 255],
                 [255, 255, 255, 255, 0, 0, 255, 255, 255, 255],
                 [255, 255, 255, 255, 255, 255, 0, 255, 255, 255],
                 [255, 255, 255, 255, 255, 255, 0, 255, 255, 255],
                 [255, 255, 255, 255, 255, 255, 255, 0, 255, 255],
                 [255, 255, 255, 255, 255, 255, 255, 255, 0, 255],
                 [255, 255, 255, 255, 255, 255, 255, 255, 255, 0],
                 [255, 255, 255, 255, 255, 255, 255, 255, 255, 0]],
                dtype=np.uint8
            )
        )

        tile = compute_rgba_tile(*args, **kwargs, format='png')
        self.assertIsInstance(tile, bytes)

    def test_compute_rgba_tile_with_components(self):
        crs_name = WEB_MERCATOR_CRS_NAME
        ml_ds = self._get_ml_dataset(crs_name)
        args = [
            ml_ds,
            ('var_a', 'var_b', 'var_c'),
            0, 0, 0
        ]
        kwargs = dict(
            crs_name=crs_name,
            tile_size=10,
            value_ranges=(0, 10),
            non_spatial_labels={'time': 0},
            tile_enlargement=0
        )
        tile = compute_rgba_tile(*args, **kwargs, format='numpy')
        self.assertIsInstance(tile, np.ndarray)
        self.assertEqual(np.uint8, tile.dtype)
        self.assertEqual((10, 10, 4), tile.shape)
        r = tile[:, :, 0]
        g = tile[:, :, 1]
        b = tile[:, :, 2]
        a = tile[:, :, 3]
        np.testing.assert_equal(
            r,
            np.array(
                [[0, 0, 76, 76, 76, 76, 102, 102, 102, 128],
                 [76, 76, 0, 76, 76, 76, 102, 102, 128, 102],
                 [76, 76, 76, 0, 76, 76, 102, 128, 102, 102],
                 [76, 76, 76, 76, 0, 0, 128, 102, 102, 102],
                 [25, 25, 25, 25, 128, 128, 0, 51, 51, 51],
                 [25, 25, 25, 25, 128, 128, 0, 51, 51, 51],
                 [25, 25, 25, 128, 25, 25, 51, 0, 51, 51],
                 [25, 25, 128, 25, 25, 25, 51, 51, 0, 51],
                 [128, 128, 25, 25, 25, 25, 51, 51, 51, 0],
                 [128, 128, 25, 25, 25, 25, 51, 51, 51, 0]],
                dtype=np.uint8
            )
        )
        np.testing.assert_equal(
            g,
            np.array(
                [[179, 179, 102, 102, 102, 102, 128, 128, 128, 0],
                 [102, 102, 179, 102, 102, 102, 128, 128, 0, 128],
                 [102, 102, 102, 179, 102, 102, 128, 0, 128, 128],
                 [102, 102, 102, 102, 179, 179, 0, 128, 128, 128],
                 [51, 51, 51, 51, 0, 0, 179, 76, 76, 76],
                 [51, 51, 51, 51, 0, 0, 179, 76, 76, 76],
                 [51, 51, 51, 0, 51, 51, 76, 179, 76, 76],
                 [51, 51, 0, 51, 51, 51, 76, 76, 179, 76],
                 [0, 0, 51, 51, 51, 51, 76, 76, 76, 179],
                 [0, 0, 51, 51, 51, 51, 76, 76, 76, 179]],
                dtype=np.uint8
            )
        )
        np.testing.assert_equal(
            b,
            np.array(
                [[204, 204, 128, 128, 128, 128, 0, 0, 0, 179],
                 [128, 128, 204, 128, 128, 128, 0, 0, 179, 0],
                 [128, 128, 128, 204, 128, 128, 0, 179, 0, 0],
                 [128, 128, 128, 128, 204, 204, 179, 0, 0, 0],
                 [76, 76, 76, 76, 179, 179, 204, 102, 102, 102],
                 [76, 76, 76, 76, 179, 179, 204, 102, 102, 102],
                 [76, 76, 76, 179, 76, 76, 102, 204, 102, 102],
                 [76, 76, 179, 76, 76, 76, 102, 102, 204, 102],
                 [179, 179, 76, 76, 76, 76, 102, 102, 102, 204],
                 [179, 179, 76, 76, 76, 76, 102, 102, 102, 204]],
                dtype=np.uint8
            )
        )
        np.testing.assert_equal(
            a,
            np.array(
                [[0, 0, 255, 255, 255, 255, 0, 0, 0, 0],
                 [255, 255, 0, 255, 255, 255, 0, 0, 0, 0],
                 [255, 255, 255, 0, 255, 255, 0, 0, 0, 0],
                 [255, 255, 255, 255, 0, 0, 0, 0, 0, 0],
                 [255, 255, 255, 255, 0, 0, 0, 255, 255, 255],
                 [255, 255, 255, 255, 0, 0, 0, 255, 255, 255],
                 [255, 255, 255, 0, 255, 255, 255, 0, 255, 255],
                 [255, 255, 0, 255, 255, 255, 255, 255, 0, 255],
                 [0, 0, 255, 255, 255, 255, 255, 255, 255, 0],
                 [0, 0, 255, 255, 255, 255, 255, 255, 255, 0]],
                dtype=np.uint8
            )
        )

        tile = compute_rgba_tile(*args, **kwargs, format='png')
        self.assertIsInstance(tile, bytes)
