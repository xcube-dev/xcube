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

import json

from ..helpers import RoutesTestCase


class PlacesRoutesTest(RoutesTestCase):

    def test_places(self):
        result, status = self.fetch_json('/places')
        self.assertEqual(200, status)
        for pg in result['placeGroups']:
            pg['sourcePaths'] = []
        self.assertEqual(
            {'placeGroups': [
                {'features': None,
                 'id': 'inside-cube',
                 'join': None,
                 'propertyMapping': None,
                 'sourceEncoding': 'utf-8',
                 'sourcePaths': [],
                 'title': 'Points inside the cube',
                 'type': 'FeatureCollection'},
                {'features': None,
                 'id': 'outside-cube',
                 'join': None,
                 'propertyMapping': None,
                 'sourceEncoding': 'utf-8',
                 'sourcePaths': [],
                 'title': 'Points outside the cube',
                 'type': 'FeatureCollection'}
            ]},
            result
        )

    def test_place_by_id(self):
        expected = {
            'places': {
                'features': [
                    {'geometry': {'coordinates': [1.5, 52.1],
                                  'type': 'Point'},
                     'id': '0',
                     'properties': {'ID': '1',
                                    'Name': 'Station 1',
                                    'Region_Name': 'Belgium',
                                    'Sub_Region_Name': 'Inside'},
                     'type': 'Feature'},
                    {'geometry': {'coordinates': [2.5, 51.5],
                                  'type': 'Point'},
                     'id': '1',
                     'properties': {'ID': '2',
                                    'Name': 'Station 2',
                                    'Region_Name': 'Belgium',
                                    'Sub_Region_Name': 'Inside'},
                     'type': 'Feature'},
                    {'geometry': {'coordinates': [4.5, 50.9],
                                  'type': 'Point'},
                     'id': '2',
                     'properties': {'ID': '3',
                                    'Name': 'Station 3',
                                    'Region_Name': 'Belgium',
                                    'Sub_Region_Name': 'Inside'},
                     'type': 'Feature'}
                ],
                'type': 'FeatureCollection'
            }
        }

        bbox = "0,40,20,60"
        result, status = self.fetch_json(f'/places/inside-cube?bbox={bbox}')
        self.assertEqual(200, status)
        self.assertEqual(expected, result)

        geom = {
            "type": "Polygon",
            "coordinates": [[
                [0, 40],
                [0, 60],
                [20, 60],
                [20, 40],
                [0, 40]
            ]]
        }
        bbox = bytes(json.dumps(geom), 'utf-8')

        result, status = self.fetch_json('/places/inside-cube',
                                         method="POST",
                                         body=bbox)
        self.assertEqual(200, status)
        self.assertEqual(expected, result)
