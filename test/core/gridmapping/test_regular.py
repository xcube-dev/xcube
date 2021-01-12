import unittest

import numpy as np
import pyproj
import xarray as xr

from xcube.core.gridmapping import CRS_WGS84
from xcube.core.gridmapping import GridMapping
from xcube.core.gridmapping.regular import RegularGridMapping

# noinspection PyProtectedMember

GEO_CRS = pyproj.crs.CRS(4326)
NOT_A_GEO_CRS = pyproj.crs.CRS(5243)


# noinspection PyMethodMayBeStatic
class RegularGridMappingTest(unittest.TestCase):

    def test_default_props(self):
        gm = GridMapping.from_min_res((1000, 1000), (10, 53), 0.01, CRS_WGS84)
        self.assertEqual((1000, 1000), gm.size)
        self.assertEqual((1000, 1000), gm.tile_size)
        self.assertEqual(10, gm.x_min)
        self.assertEqual(53, gm.y_min)
        self.assertEqual((0.01, 0.01), gm.xy_res)
        self.assertEqual(True, gm.is_regular)
        self.assertEqual(False, gm.is_j_axis_up)

    def test_invalid_y(self):
        with self.assertRaises(ValueError) as cm:
            GridMapping.from_min_res((1000, 1000), (10, -90.5), 0.01, CRS_WGS84)
        self.assertEqual('invalid y_min', f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            GridMapping.from_min_res((1000, 1000), (10, 53), 0.1, CRS_WGS84)
        self.assertEqual('invalid size, y_min combination', f'{cm.exception}')

    def test_xy_bbox(self):
        gm = GridMapping.from_min_res((1000, 1000), (10, 53), 0.01, CRS_WGS84)
        self.assertEqual((10, 53, 20, 63), gm.xy_bbox)
        self.assertEqual(False, gm.is_lon_360)

    def test_xy_bbox_anti_meridian(self):
        gm = GridMapping.from_min_res((2000, 1000), (174.0, -30.0), 0.005, CRS_WGS84)
        self.assertEqual((174.0, -30.0, 184.0, -25.0), gm.xy_bbox)
        self.assertEqual(True, gm.is_lon_360)

    def test_derive(self):
        gm = GridMapping.from_min_res((1000, 1000), (10, 53), 0.01, CRS_WGS84)
        self.assertEqual((1000, 1000), gm.size)
        self.assertEqual((1000, 1000), gm.tile_size)
        self.assertEqual(False, gm.is_j_axis_up)
        derived_gm = gm.derive(tile_size=500, is_j_axis_up=True)
        self.assertIsNot(gm, derived_gm)
        self.assertIsInstance(derived_gm, RegularGridMapping)
        self.assertEqual((1000, 1000), derived_gm.size)
        self.assertEqual((500, 500), derived_gm.tile_size)
        self.assertEqual(True, derived_gm.is_j_axis_up)

    def test_xy_coords(self):
        gm = GridMapping.from_min_res((1000, 1000), (10, 53), 0.01, CRS_WGS84, tile_size=(500, 500))
        xy_coords = gm.xy_coords
        self.assertIsInstance(xy_coords, xr.DataArray)
        self.assertIs(gm.xy_coords, xy_coords)
        self.assertEqual(('coord', 'lat', 'lon'), xy_coords.dims)
        self.assertEqual((2, 1000, 1000), xy_coords.shape)
        self.assertEqual(((2,), (500, 500), (500, 500)), xy_coords.chunks)

    def test_ij_bboxes(self):
        gm = GridMapping.from_min_res(size=(2000, 1000),
                                      xy_min=(10.0, 20.0), xy_res=0.1, crs=NOT_A_GEO_CRS)
        np.testing.assert_almost_equal(gm.ij_bboxes,
                                       np.array([[0, 0, 1999, 999]], dtype=np.int64))

        image_geom = GridMapping.from_min_res(size=(2000, 1000),
                                              xy_min=(10.0, 20.0), xy_res=0.1, tile_size=500, crs=NOT_A_GEO_CRS)
        np.testing.assert_almost_equal(image_geom.ij_bboxes,
                                       np.array([
                                           [0, 0, 499, 499],
                                           [500, 0, 999, 499],
                                           [1000, 0, 1499, 499],
                                           [1500, 0, 1999, 499],
                                           [0, 500, 499, 999],
                                           [500, 500, 999, 999],
                                           [1000, 500, 1499, 999],
                                           [1500, 500, 1999, 999]
                                       ], dtype=np.int64))

    def test_xy_bboxes(self):
        image_geom = GridMapping.from_min_res(size=(2000, 1000),
                                              xy_min=(10.0, 20.0), xy_res=0.1, crs=NOT_A_GEO_CRS)
        np.testing.assert_almost_equal(image_geom.xy_bboxes,
                                       np.array([[10., 20.1, 209.9, 120.]], dtype=np.float64))

        image_geom = GridMapping.from_min_res(size=(2000, 1000),
                                              xy_min=(10.0, 20.0), xy_res=0.1, tile_size=500, crs=NOT_A_GEO_CRS)
        np.testing.assert_almost_equal(image_geom.xy_bboxes,
                                       np.array([
                                           [10., 70.1, 59.9, 120.],
                                           [60., 70.1, 109.9, 120.],
                                           [110., 70.1, 159.9, 120.],
                                           [160., 70.1, 209.9, 120.],
                                           [10., 20.1, 59.9, 70.],
                                           [60., 20.1, 109.9, 70.],
                                           [110., 20.1, 159.9, 70.],
                                           [160., 20.1, 209.9, 70.]
                                       ], dtype=np.float64))

    def test_xy_bboxes_is_j_axis_up(self):
        image_geom = GridMapping.from_min_res(size=(2000, 1000), is_j_axis_up=True,
                                              xy_min=(10.0, 20.0), xy_res=0.1, crs=NOT_A_GEO_CRS)
        np.testing.assert_almost_equal(image_geom.xy_bboxes,
                                       np.array([[10., 20., 209.9, 119.9]], dtype=np.float64))

        image_geom = GridMapping.from_min_res(size=(2000, 1000), is_j_axis_up=True,
                                              xy_min=(10.0, 20.0), xy_res=0.1, crs=NOT_A_GEO_CRS,
                                              tile_size=500)
        np.testing.assert_almost_equal(image_geom.xy_bboxes,
                                       np.array([
                                           [10., 20., 59.9, 69.9],
                                           [60., 20., 109.9, 69.9],
                                           [110., 20., 159.9, 69.9],
                                           [160., 20., 209.9, 69.9],
                                           [10., 70., 59.9, 119.9],
                                           [60., 70., 109.9, 119.9],
                                           [110., 70., 159.9, 119.9],
                                           [160., 70., 209.9, 119.9]
                                       ], dtype=np.float64))

    def test_coord_vars(self):
        image_geom = GridMapping.from_min_res(size=(10, 6), xy_min=(-2600.0, 1200.0), xy_res=10.0, crs=NOT_A_GEO_CRS)

        cv = image_geom.coord_vars(xy_names=('x', 'y'))
        self._assert_coord_vars(cv,
                                (10, 6),
                                ('x', 'y'),
                                (-2595., -2505.),
                                (1255., 1205.),
                                ('x_bnds', 'y_bnds'),
                                (
                                    (-2600., -2590.),
                                    (-2510., -2500.),
                                ),
                                (
                                    (1250., 1260.),
                                    (1200., 1210.),
                                ))

    def test_coord_vars_j_axis_up(self):
        image_geom = GridMapping.from_min_res(size=(10, 6), xy_min=(-2600.0, 1200.0), xy_res=10.0,
                                              is_j_axis_up=True, crs=NOT_A_GEO_CRS)

        cv = image_geom.coord_vars(xy_names=('x', 'y'))
        self._assert_coord_vars(cv,
                                (10, 6),
                                ('x', 'y'),
                                (-2595., -2505.),
                                (1205., 1255.),
                                ('x_bnds', 'y_bnds'),
                                (
                                    (-2600., -2590.),
                                    (-2510., -2500.),
                                ),
                                (
                                    (1200., 1210.),
                                    (1250., 1260.),
                                ))

    def test_coord_vars_antimeridian(self):
        image_geom = GridMapping.from_min_res(size=(10, 10), xy_min=(172.0, 53.0), xy_res=2.0, crs=GEO_CRS)

        cv = image_geom.coord_vars(xy_names=('lon', 'lat'))
        self._assert_coord_vars(cv,
                                (10, 10),
                                ('lon', 'lat'),
                                (173.0, -169.0),
                                (72.0, 54.0),
                                ('lon_bnds', 'lat_bnds'),
                                (
                                    (172., 174.),
                                    (-170., -168.),
                                ),
                                (
                                    (71., 73.),
                                    (53., 55.),
                                ))

    def _assert_coord_vars(self,
                           cv,
                           size,
                           xy_names,
                           x_values,
                           y_values,
                           xy_bnds_names,
                           x_bnds_values,
                           y_bnds_values):
        self.assertIsNotNone(cv)
        self.assertIn(xy_names[0], cv)
        self.assertIn(xy_names[1], cv)
        self.assertIn(xy_bnds_names[0], cv)
        self.assertIn(xy_bnds_names[1], cv)

        x = cv[xy_names[0]]
        self.assertEqual((size[0],), x.shape)
        np.testing.assert_almost_equal(x.values[0], np.array(x_values[0]))
        np.testing.assert_almost_equal(x.values[-1], np.array(x_values[-1]))

        y = cv[xy_names[1]]
        self.assertEqual((size[1],), y.shape)
        np.testing.assert_almost_equal(y.values[0], np.array(y_values[0]))
        np.testing.assert_almost_equal(y.values[-1], np.array(y_values[-1]))

        x_bnds = cv[xy_bnds_names[0]]
        self.assertEqual((size[0], 2), x_bnds.shape)
        np.testing.assert_almost_equal(x_bnds.values[0], np.array(x_bnds_values[0]))
        np.testing.assert_almost_equal(x_bnds.values[-1], np.array(x_bnds_values[-1]))

        y_bnds = cv[xy_bnds_names[1]]
        self.assertEqual((size[1], 2), y_bnds.shape)
        np.testing.assert_almost_equal(y_bnds.values[0], y_bnds_values[0])
        np.testing.assert_almost_equal(y_bnds.values[-1], y_bnds_values[-1])
