# The MIT License (MIT)
# Copyright (c) 2022 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from ..helpers import RoutesTestCase


class TimeSeriesRoutesTest(RoutesTestCase):

    def test_fetch_timeseries_invalid_body(self):
        response = self.fetch('/timeseries/demo/conc_chl', method="POST",
                              body='')
        self.assertBadRequestResponse(
            response,
            'HTTP 400:'
            ' Bad Request ('
            'Body does not contain valid JSON:'
            ' Expecting value: line 1 column 1 (char 0)'
            ')'
        )

        response = self.fetch(
            '/timeseries/demo/conc_chl', method="POST",
            body='{"type":"Point"}'
        )
        self.assertBadRequestResponse(
            response,
            'GeoJSON object expected'
        )

    def test_fetch_timeseries_geometry(self):
        response = self.fetch(
            '/timeseries/demo/conc_chl', method="POST",
            body='{"type": "Point", "coordinates": [1, 51]}'
        )
        self.assertResponseOK(response)

        response = self.fetch(
            '/timeseries/demo/conc_chl', method="POST",
            body='{"type":"Polygon",'
                 ' "coordinates": [[[1, 51], [2, 51], [2, 52], [1, 51]]]}'
        )
        self.assertResponseOK(response)

    def test_fetch_timeseries_geometry_collection(self):
        response = self.fetch(
            '/timeseries/demo/conc_chl', method="POST",
            body='{"type": "GeometryCollection", "geometries": null}'
        )
        self.assertResponseOK(response)

        response = self.fetch(
            '/timeseries/demo/conc_chl', method="POST",
            body='{"type": "GeometryCollection", "geometries": []}'
        )
        self.assertResponseOK(response)

        response = self.fetch(
            '/timeseries/demo/conc_chl', method="POST",
            body='{"type": "GeometryCollection", "geometries": '
                 '[{"type": "Point", "coordinates": [1, 51]}]}'
        )
        self.assertResponseOK(response)

    def test_fetch_timeseries_feature(self):
        response = self.fetch(
            '/timeseries/demo/conc_chl', method="POST",
            body='{"type": "Feature", '
                 ' "properties": {}, '
                 ' "geometry": {"type": "Point", "coordinates": [1, 51]}'
                 '}'
        )
        self.assertResponseOK(response)

    def test_fetch_timeseries_feature_collection(self):
        response = self.fetch(
            '/timeseries/demo/conc_chl', method="POST",
            body='{"type": "FeatureCollection", "features": null}'
        )
        self.assertResponseOK(response)

        response = self.fetch(
            '/timeseries/demo/conc_chl', method="POST",
            body='{"type": "FeatureCollection", "features": []}'
        )
        self.assertResponseOK(response)

        response = self.fetch(
            '/timeseries/demo/conc_chl', method="POST",
            body='{"type": "FeatureCollection", "features": ['
                 '  {"type": "Feature", "properties": {}, '
                 '   "geometry": {"type": "Point", "coordinates": [1, 51]}}'
                 ']}'
        )
        self.assertResponseOK(response)

    def test_fetch_timeseries_tolerance(self):
        response = self.fetch(
            '/timeseries/demo/conc_chl?tolerance=60',
            method="POST",
            body='{"type": "Point", "coordinates": [1, 51]}'
        )
        self.assertResponseOK(response)

        response = self.fetch(
            '/timeseries/demo/conc_chl?tolerance=2D',
            method="POST",
            body='{"type": "Point", "coordinates": [1, 51]}'
        )
        self.assertBadRequestResponse(
            response,
            'HTTP 400:'
            ' Bad Request ('
            "Query parameter 'tolerance' must have type 'float'."
            ')'
        )
