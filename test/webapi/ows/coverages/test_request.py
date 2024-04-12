# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest
import pyproj

from xcube.webapi.ows.coverages.request import CoverageRequest


class CoveragesRequestTest(unittest.TestCase):
    def test_parse_bbox(self):
        self.assertIsNone(CoverageRequest({}).bbox)
        self.assertEqual(
            [1.1, 2.2, 3.3, 4.4],
            CoverageRequest(dict(bbox=["1.1,2.2,3.3,4.4"])).bbox,
        )
        with self.assertRaises(ValueError):
            CoverageRequest(dict(bbox=["foo,bar,baz"]))
        with self.assertRaises(ValueError):
            CoverageRequest(dict(bbox=["1.1,2.2,3.3"]))

    def test_parse_bbox_crs(self):
        self.assertEqual(
            pyproj.CRS("OGC:CRS84"),
            CoverageRequest({}).bbox_crs,
        )
        self.assertEqual(
            pyproj.CRS(crs_spec := "EPSG:4326"),
            CoverageRequest({"bbox-crs": [crs_spec]}).bbox_crs,
        )
        self.assertEqual(
            pyproj.CRS(crs_spec := "OGC:CRS84"),
            CoverageRequest({"bbox-crs": [f"[{crs_spec}]"]}).bbox_crs,
        )
        with self.assertRaises(ValueError):
            CoverageRequest({"bbox-crs": ["not a CRS specifier"]})

    def test_parse_datetime(self):
        dt0 = "2018-02-12T23:20:52Z"
        dt1 = "2019-02-12T23:20:52Z"
        self.assertIsNone(CoverageRequest({}).datetime)
        self.assertEqual(dt0, CoverageRequest({"datetime": [dt0]}).datetime)
        self.assertEqual(
            (dt0, None), CoverageRequest({"datetime": [f"{dt0}/.."]}).datetime
        )
        self.assertEqual(
            (None, dt1), CoverageRequest({"datetime": [f"../{dt1}"]}).datetime
        )
        self.assertEqual(
            (dt0, dt1),
            CoverageRequest({"datetime": [f"{dt0}/{dt1}"]}).datetime,
        )
        with self.assertRaises(ValueError):
            CoverageRequest({"datetime": [f"{dt0}/{dt0}/{dt1}"]})
        with self.assertRaises(ValueError):
            CoverageRequest({"datetime": ["not a valid time string"]})

    def test_parse_subset(self):
        self.assertIsNone(CoverageRequest({}).subset)
        self.assertEqual(
            dict(Lat=("10", "20"), Lon=("30", None), time="2019-03-27"),
            CoverageRequest(
                dict(subset=['Lat(10:20),Lon(30:*),time("2019-03-27")'])
            ).subset,
        )
        self.assertEqual(
            dict(Lat=(None, "20"), Lon="30", time=("2019-03-27", "2020-03-27")),
            CoverageRequest(
                dict(subset=['Lat(*:20),Lon(30),time("2019-03-27":"2020-03-27")'])
            ).subset,
        )
        with self.assertRaises(ValueError):
            CoverageRequest({"subset": ["not a valid specifier"]})

    def test_parse_subset_crs(self):
        self.assertEqual(
            pyproj.CRS("OGC:CRS84"),
            CoverageRequest({}).subset_crs,
        )
        self.assertEqual(
            pyproj.CRS(crs_spec := "EPSG:4326"),
            CoverageRequest({"subset-crs": [crs_spec]}).subset_crs,
        )
        with self.assertRaises(ValueError):
            CoverageRequest({"subset-crs": ["not a CRS specifier"]})

    def test_parse_properties(self):
        self.assertIsNone(CoverageRequest({}).properties)
        self.assertEqual(
            ["foo", "bar", "baz"],
            CoverageRequest(dict(properties=["foo,bar,baz"])).properties,
        )

    def test_parse_scale_factor(self):
        self.assertEqual(None, CoverageRequest({}).scale_factor)
        self.assertEqual(1.5, CoverageRequest({"scale-factor": ["1.5"]}).scale_factor)
        with self.assertRaises(ValueError):
            CoverageRequest({"scale-factor": ["this is not a number"]})

    def test_parse_scale_axes(self):
        self.assertIsNone(CoverageRequest({}).scale_axes)
        self.assertEqual(
            dict(Lat=1.5, Lon=2.5),
            CoverageRequest({"scale-axes": ["Lat(1.5),Lon(2.5)"]}).scale_axes,
        )
        with self.assertRaises(ValueError):
            CoverageRequest({"scale-axes": ["Lat(1.5"]})
        with self.assertRaises(ValueError):
            CoverageRequest({"scale-axes": ["Lat(not a number)"]})

    def test_parse_scale_size(self):
        self.assertIsNone(CoverageRequest({}).scale_size)
        self.assertEqual(
            dict(Lat=12.3, Lon=45.6),
            CoverageRequest({"scale-size": ["Lat(12.3),Lon(45.6)"]}).scale_size,
        )
        with self.assertRaises(ValueError):
            CoverageRequest({"scale-size": ["Lat(1.5"]})
        with self.assertRaises(ValueError):
            CoverageRequest({"scale-size": ["Lat(not a number)"]})

    def test_parse_crs(self):
        self.assertIsNone(CoverageRequest({}).crs)
        self.assertEqual(
            pyproj.CRS(crs := "EPSG:4326"), CoverageRequest({"crs": [crs]}).crs
        )
        with self.assertRaises(ValueError):
            CoverageRequest({"crs": ["an invalid CRS specifier"]})
