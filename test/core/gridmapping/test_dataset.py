import os.path
import unittest

import numpy as np
import pyproj
import xarray as xr

import xcube.core.new
from xcube.core.gridmapping import GridMapping

# noinspection PyProtectedMember

GEO_CRS = pyproj.crs.CRS(4326)
NOT_A_GEO_CRS = pyproj.crs.CRS(5243)

OLCI_L2_PATH = os.path.join(os.path.dirname(__file__),
                            '..', '..', '..',
                            'examples', 'notebooks', 'S3-OLCI-L2A.zarr.zip')


# noinspection PyMethodMayBeStatic
class DatasetGridMappingTest(unittest.TestCase):
    def test_from_regular_cube(self):
        dataset = xcube.core.new.new_cube(variables=dict(rad=0.5))
        gm = GridMapping.from_dataset(dataset)
        self.assertEqual((360, 180), gm.size)
        self.assertEqual((360, 180), gm.tile_size)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual((1, 1), gm.xy_res)
        self.assertEqual(True, gm.is_regular)
        self.assertEqual(False, gm.is_lon_360)
        self.assertEqual(True, gm.is_j_axis_up)
        self.assertEqual((2, 180, 360), gm.xy_coords.shape)
        self.assertEqual(('coord', 'lat', 'lon'), gm.xy_coords.dims)

    def test_from_non_regular_cube(self):
        lon = np.array([[8, 9.3, 10.6, 11.9],
                        [8, 9.2, 10.4, 11.6],
                        [8, 9.1, 10.2, 11.3]], dtype=np.float32)
        lat = np.array([[56, 56.1, 56.2, 56.3],
                        [55, 55.2, 55.4, 55.6],
                        [54, 54.3, 54.6, 54.9]], dtype=np.float32)
        rad = np.random.random(3 * 4).reshape((3, 4))
        dims = ('y', 'x')
        dataset = xr.Dataset(dict(lon=xr.DataArray(lon, dims=dims),
                                  lat=xr.DataArray(lat, dims=dims),
                                  rad=xr.DataArray(rad, dims=dims)))
        gm = GridMapping.from_dataset(dataset)
        self.assertEqual((4, 3), gm.size)
        self.assertEqual((4, 3), gm.tile_size)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual(False, gm.is_regular)
        self.assertEqual(False, gm.is_lon_360)
        self.assertEqual(None, gm.is_j_axis_up)
        self.assertEqual((2, 3, 4), gm.xy_coords.shape)
        self.assertEqual(('coord', 'y', 'x'), gm.xy_coords.dims)
        self.assertEqual((0.93, 0.93), gm.xy_res)

    def test_from_real_olci(self):
        dataset = xr.open_zarr(OLCI_L2_PATH)
        gm = GridMapping.from_dataset(dataset)
        self.assertEqual((1189, 1890), gm.size)
        self.assertEqual((512, 512), gm.tile_size)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual((0.00098, 0.00098), gm.xy_res)
        # self.assertAlmostEqual(12.693771178309552, gm.x_min)
        # self.assertAlmostEqual(20.005413821690446, gm.x_max)
        # self.assertAlmostEqual(55.19965017830955, gm.y_min)
        # self.assertAlmostEqual(60.63871982169044, gm.y_max)
        self.assertEqual(False, gm.is_regular)
        self.assertEqual(False, gm.is_lon_360)
        self.assertEqual(None, gm.is_j_axis_up)
        self.assertEqual((2, 1890, 1189), gm.xy_coords.shape)
        self.assertEqual(('coord', 'y', 'x'), gm.xy_coords.dims)

        gm = gm.to_regular()
        self.assertEqual((7462, 5551), gm.size)
