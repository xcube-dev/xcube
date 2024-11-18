# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from ..helpers import RoutesTestCase


class TimeSeriesRoutesTest(RoutesTestCase):
    def test_fetch_timeseries_invalid_body(self):
        response = self.fetch("/timeseries/demo/conc_chl", method="POST", body="")
        self.assertBadRequestResponse(
            response,
            "HTTP 400:"
            " Bad Request ("
            "Body does not contain valid JSON:"
            " Expecting value: line 1 column 1 (char 0)"
            ")",
        )

        response = self.fetch(
            "/timeseries/demo/conc_chl", method="POST", body='{"type":"Point"}'
        )
        self.assertBadRequestResponse(response, "GeoJSON object expected")

    def test_fetch_timeseries_geometry(self):
        response = self.fetch(
            "/timeseries/demo/conc_chl",
            method="POST",
            body='{"type": "Point", "coordinates": [1, 51]}',
        )
        self.assertResponseOK(response)

        response = self.fetch(
            "/timeseries/demo/conc_chl",
            method="POST",
            body='{"type":"Polygon",'
            ' "coordinates": [[[1, 51], [2, 51], [2, 52], [1, 51]]]}',
        )
        self.assertResponseOK(response)

    def test_fetch_timeseries_geometry_collection(self):
        response = self.fetch(
            "/timeseries/demo/conc_chl",
            method="POST",
            body='{"type": "GeometryCollection", "geometries": null}',
        )
        self.assertResponseOK(response)

        response = self.fetch(
            "/timeseries/demo/conc_chl",
            method="POST",
            body='{"type": "GeometryCollection", "geometries": []}',
        )
        self.assertResponseOK(response)

        response = self.fetch(
            "/timeseries/demo/conc_chl",
            method="POST",
            body='{"type": "GeometryCollection", "geometries": '
            '[{"type": "Point", "coordinates": [1, 51]}]}',
        )
        self.assertResponseOK(response)

    def test_fetch_timeseries_feature(self):
        response = self.fetch(
            "/timeseries/demo/conc_chl",
            method="POST",
            body='{"type": "Feature", '
            ' "properties": {}, '
            ' "geometry": {"type": "Point", "coordinates": [1, 51]}'
            "}",
        )
        self.assertResponseOK(response)

    def test_fetch_timeseries_feature_collection(self):
        response = self.fetch(
            "/timeseries/demo/conc_chl",
            method="POST",
            body='{"type": "FeatureCollection", "features": null}',
        )
        self.assertResponseOK(response)

        response = self.fetch(
            "/timeseries/demo/conc_chl",
            method="POST",
            body='{"type": "FeatureCollection", "features": []}',
        )
        self.assertResponseOK(response)

        response = self.fetch(
            "/timeseries/demo/conc_chl",
            method="POST",
            body='{"type": "FeatureCollection", "features": ['
            '  {"type": "Feature", "properties": {}, '
            '   "geometry": {"type": "Point", "coordinates": [1, 51]}}'
            "]}",
        )
        self.assertResponseOK(response)

    def test_fetch_timeseries_tolerance(self):
        response = self.fetch(
            "/timeseries/demo/conc_chl?tolerance=60",
            method="POST",
            body='{"type": "Point", "coordinates": [1, 51]}',
        )
        self.assertResponseOK(response)

        response = self.fetch(
            "/timeseries/demo/conc_chl?tolerance=2D",
            method="POST",
            body='{"type": "Point", "coordinates": [1, 51]}',
        )
        self.assertBadRequestResponse(
            response,
            "HTTP 400:"
            " Bad Request ("
            "Query parameter 'tolerance' must have type 'float'."
            ")",
        )
