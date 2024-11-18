# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

from xcube.server.api import ApiError
from xcube.webapi.places.controllers import find_places
from .test_context import get_places_ctx

BAD_REQUEST_MSG = (
    "HTTP status 400:"
    " Received invalid geometry bbox,"
    " geometry WKT, or GeoJSON object"
)


class PlacesControllersTest(unittest.TestCase):
    def test_find_places_all(self):
        ctx = get_places_ctx()
        places = find_places(ctx, "all", "http://localhost:9090")
        self._assertPlaceGroup(places, 6, {"0", "1", "2", "3", "4", "5"})

    def test_find_places_by_box_ok(self):
        ctx = get_places_ctx()
        places = find_places(
            ctx, "all", "http://localhost:9090", query_geometry="-1,49,2,55"
        )
        self._assertPlaceGroup(places, 2, {"0", "3"})

    def test_find_places_by_box_fail(self):
        ctx = get_places_ctx()

        with self.assertRaises(ApiError.BadRequest) as cm:
            find_places(ctx, "all", "http://localhost:9090", query_geometry="-1,49,55")
        self.assertEqual(BAD_REQUEST_MSG, f"{cm.exception}")

        with self.assertRaises(ApiError.BadRequest) as cm:
            find_places(
                ctx, "all", "http://localhost:9090", query_geometry="-1,49,x,55"
            )
        self.assertEqual(BAD_REQUEST_MSG, f"{cm.exception}")

    def test_find_places_by_wkt_ok(self):
        ctx = get_places_ctx()
        places = find_places(
            ctx,
            "all",
            "http://localhost:9090",
            query_geometry="POLYGON ((2 49, 2 55, -1 55, -1 49, 2 49))",
        )
        self._assertPlaceGroup(places, 2, {"0", "3"})

    def test_find_places_by_wkt_fail(self):
        ctx = get_places_ctx()

        with self.assertRaises(ApiError.BadRequest) as cm:
            find_places(
                ctx,
                "all",
                "http://localhost:9090",
                query_geometry="POLYGLON ((2 49, 2 55, -1 55, -1 49, 2 49))",
            )
        self.assertEqual(BAD_REQUEST_MSG, f"{cm.exception}")

    def test_find_places_by_geojson_ok(self):
        ctx = get_places_ctx()

        geojson_obj = {
            "type": "Polygon",
            "coordinates": (
                ((2.0, 49.0), (2.0, 55.0), (-1.0, 55.0), (-1.0, 49.0), (2.0, 49.0)),
            ),
        }
        places = find_places(
            ctx, "all", "http://localhost:9090", query_geometry=geojson_obj
        )
        self._assertPlaceGroup(places, 2, {"0", "3"})

        geojson_obj = {"type": "Feature", "properties": {}, "geometry": geojson_obj}
        places = find_places(
            ctx, "all", "http://localhost:9090", query_geometry=geojson_obj
        )
        self._assertPlaceGroup(places, 2, {"0", "3"})

        geojson_obj = {"type": "FeatureCollection", "features": [geojson_obj]}
        places = find_places(
            ctx, "all", "http://localhost:9090", query_geometry=geojson_obj
        )
        self._assertPlaceGroup(places, 2, {"0", "3"})

    def test_find_places_by_geojson_fail(self):
        ctx = get_places_ctx()
        with self.assertRaises(ApiError.BadRequest) as cm:
            geojson_obj = {"type": "FeatureCollection", "features": []}
            find_places(ctx, "all", "http://localhost:9090", query_geometry=geojson_obj)
        self.assertEqual(BAD_REQUEST_MSG, f"{cm.exception}")

    def _assertPlaceGroup(self, feature_collection, expected_count, expected_ids):
        self.assertIsInstance(feature_collection, dict)
        self.assertIn("type", feature_collection)
        self.assertEqual("FeatureCollection", feature_collection["type"])
        self.assertIn("features", feature_collection)
        features = feature_collection["features"]
        self.assertIsInstance(features, list)
        self.assertEqual(expected_count, len(features))
        actual_ids = {f["id"] for f in features if "id" in f}
        self.assertEqual(expected_ids, actual_ids)
