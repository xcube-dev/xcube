# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest
from dataclasses import dataclass

import pyproj

import xcube.core.new
from xcube.server.api import ApiError
from xcube.webapi.ows.coverages.request import CoverageRequest
from xcube.webapi.ows.coverages.scaling import CoverageScaling


class ScalingTest(unittest.TestCase):
    def setUp(self):
        self.epsg4326 = pyproj.CRS("EPSG:4326")
        self.ds = xcube.core.new.new_cube()

    def test_default_scaling(self):
        scaling = CoverageScaling(CoverageRequest({}), self.epsg4326, self.ds)
        self.assertEqual((1, 1), scaling.factor)

    def test_no_data(self):
        with self.assertRaises(ApiError.NotFound):
            CoverageScaling(
                CoverageRequest({}),
                self.epsg4326,
                self.ds.isel(lat=slice(0, 0)),
            )

    def test_crs_no_valid_axis(self):
        @dataclass
        class CrsMock:
            axis_info = [object()]

        # noinspection PyTypeChecker
        self.assertIsNone(
            CoverageScaling(CoverageRequest({}), CrsMock(), self.ds).get_axis_from_crs(
                set()
            )
        )

    def test_scale_factor(self):
        scaling = CoverageScaling(
            CoverageRequest({"scale-factor": ["2"]}), self.epsg4326, self.ds
        )
        self.assertEqual((2, 2), scaling.factor)
        self.assertEqual((180, 90), scaling.size)

    def test_scale_axes(self):
        scaling = CoverageScaling(
            CoverageRequest({"scale-axes": ["Lat(3),Lon(1.2)"]}),
            self.epsg4326,
            self.ds,
        )
        self.assertEqual((1.2, 3), scaling.factor)
        self.assertEqual((300, 60), scaling.size)

    def test_scale_size(self):
        scaling = CoverageScaling(
            CoverageRequest({"scale-size": ["Lat(90),Lon(240)"]}),
            self.epsg4326,
            self.ds,
        )
        self.assertEqual((240, 90), scaling.size)
        self.assertEqual((1.5, 2), scaling.factor)

    def test_apply_identity_scaling(self):
        # noinspection PyTypeChecker
        self.assertEqual(
            gm_mock := object(),
            CoverageScaling(
                CoverageRequest({"scale-factor": ["1"]}),
                self.epsg4326,
                self.ds,
            ).apply(gm_mock),
        )
