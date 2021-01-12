import os.path
import unittest

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
        src_ds = xcube.core.new.new_cube()
        gm = GridMapping.from_dataset(src_ds)
        self.assertEqual((360, 180), gm.size)
        self.assertEqual((360, 180), gm.tile_size)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertEqual((1, 1), gm.xy_res)
        self.assertEqual(True, gm.is_regular)
        self.assertEqual(False, gm.is_lon_360)
        self.assertEqual(True, gm.is_j_axis_up)
        self.assertEqual((2, 180, 360), gm.xy_coords.shape)
        self.assertEqual(('coord', 'lat', 'lon'), gm.xy_coords.dims)

    def test_from_real_olci(self):
        src_ds = xr.open_zarr(OLCI_L2_PATH)
        gm = GridMapping.from_dataset(src_ds)
        self.assertEqual((1189, 1890), gm.size)
        self.assertEqual((512, 512), gm.tile_size)
        self.assertEqual(GEO_CRS, gm.crs)
        self.assertAlmostEqual(0.0010596433808944803, gm.x_res)
        self.assertAlmostEqual(0.0010596433808944803, gm.y_res)
        self.assertAlmostEqual(12.693771178309552, gm.x_min)
        self.assertAlmostEqual(20.005413821690446, gm.x_max)
        self.assertAlmostEqual(55.19965017830955, gm.y_min)
        self.assertAlmostEqual(60.63871982169044, gm.y_max)
        self.assertEqual(False, gm.is_regular)
        self.assertEqual(False, gm.is_lon_360)
        self.assertEqual(None, gm.is_j_axis_up)
        self.assertEqual((2, 1890, 1189), gm.xy_coords.shape)
        self.assertEqual(('coord', 'y', 'x'), gm.xy_coords.dims)
