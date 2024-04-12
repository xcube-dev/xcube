# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import json

from ..helpers import RoutesTestCase


class PlacesRoutesTest(RoutesTestCase):
    def test_places(self):
        result, status = self.fetch_json("/places")
        self.assertEqual(200, status)
        for pg in result["placeGroups"]:
            pg["sourcePaths"] = []
        self.assertEqual(
            {
                "placeGroups": [
                    {
                        "features": None,
                        "id": "inside-cube",
                        "join": None,
                        "propertyMapping": None,
                        "sourceEncoding": "utf-8",
                        "sourcePaths": [],
                        "title": "Points inside the cube",
                        "type": "FeatureCollection",
                    },
                    {
                        "features": None,
                        "id": "outside-cube",
                        "join": None,
                        "propertyMapping": None,
                        "sourceEncoding": "utf-8",
                        "sourcePaths": [],
                        "title": "Points outside the cube",
                        "type": "FeatureCollection",
                    },
                ]
            },
            result,
        )

    def test_place_by_id(self):
        expected = {
            "places": {
                "features": [
                    {
                        "geometry": {"coordinates": [1.5, 52.1], "type": "Point"},
                        "id": "0",
                        "properties": {
                            "ID": "1",
                            "Name": "Station 1",
                            "Region_Name": "Belgium",
                            "Sub_Region_Name": "Inside",
                        },
                        "type": "Feature",
                    },
                    {
                        "geometry": {"coordinates": [2.5, 51.5], "type": "Point"},
                        "id": "1",
                        "properties": {
                            "ID": "2",
                            "Name": "Station 2",
                            "Region_Name": "Belgium",
                            "Sub_Region_Name": "Inside",
                        },
                        "type": "Feature",
                    },
                    {
                        "geometry": {"coordinates": [4.5, 50.9], "type": "Point"},
                        "id": "2",
                        "properties": {
                            "ID": "3",
                            "Name": "Station 3",
                            "Region_Name": "Belgium",
                            "Sub_Region_Name": "Inside",
                        },
                        "type": "Feature",
                    },
                ],
                "type": "FeatureCollection",
            }
        }

        bbox = "0,40,20,60"
        result, status = self.fetch_json(f"/places/inside-cube?bbox={bbox}")
        self.assertEqual(200, status)
        self.assertEqual(expected, result)

        geom = {
            "type": "Polygon",
            "coordinates": [[[0, 40], [0, 60], [20, 60], [20, 40], [0, 40]]],
        }
        bbox = bytes(json.dumps(geom), "utf-8")

        result, status = self.fetch_json(
            "/places/inside-cube", method="POST", body=bbox
        )
        self.assertEqual(200, status)
        self.assertEqual(expected, result)
