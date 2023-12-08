import unittest

import pyproj

import xcube.core.new
from xcube.server.api import ApiError
from xcube.webapi.ows.coverages.request import CoveragesRequest
from xcube.webapi.ows.coverages.scaling import CoverageScaling


class ScalingTest(unittest.TestCase):

    def setUp(self):
        self.epsg4326 = pyproj.CRS('EPSG:4326')
        self.ds = xcube.core.new.new_cube()

    def test_default_scaling(self):
        scaling = CoverageScaling(CoveragesRequest({}), self.epsg4326, self.ds)
        self.assertEqual((1, 1), scaling.scale)

    def test_no_data(self):
        with self.assertRaises(ApiError.NotFound):
            CoverageScaling(
                CoveragesRequest({}), self.epsg4326,
                self.ds.isel(lat=slice(0, 0))
            )

    def test_crs_fallbacks(self):
        # TODO: implement me
        pass

    def test_scale_factor(self):
        scaling = CoverageScaling(
            CoveragesRequest({'scale-factor': ['2']}),
            self.epsg4326,
            self.ds
        )
        self.assertEqual((2, 2), scaling.scale)

    def test_scale_axes(self):
        scaling = CoverageScaling(
            CoveragesRequest({'scale-axes': ['Lat(3),Lon(1.2)']}),
            self.epsg4326,
            self.ds
        )
        self.assertEqual((1.2, 3), scaling.scale)
        self.assertEqual((300, 60), scaling.size)

    def test_scale_size(self):
        scaling = CoverageScaling(
            CoveragesRequest({'scale-size': ['Lat(90),Lon(240)']}),
            self.epsg4326,
            self.ds
        )
        self.assertEqual((240, 90), scaling.size)
        self.assertEqual((1.5, 2), scaling.scale)
