import unittest

from xcube.webapi.controllers.places import find_places, find_dataset_places
from xcube.webapi.errors import ServiceBadRequestError
from ..helpers import new_test_service_context


class FeaturesControllerTest(unittest.TestCase):

    def test_find_places_all(self):
        ctx = new_test_service_context()
        places = find_places(ctx, "all")
        self._assertPlaceGroup(places, 6, {'0', '1', '2', '3', '4', '5'})

    def test_find_places_by_box(self):
        ctx = new_test_service_context()
        places = find_places(ctx, "all", box_coords="-1,49,2,55")
        self._assertPlaceGroup(places, 2, {'0', '3'})

        with self.assertRaises(ServiceBadRequestError) as cm:
            find_places(ctx, "all", box_coords="-1,49,55")
        self.assertEqual("HTTP 400: Received invalid bounding box geometry", f"{cm.exception}")

        with self.assertRaises(ServiceBadRequestError) as cm:
            find_places(ctx, "all", box_coords="-1,49,x,55")
        self.assertEqual("HTTP 400: Received invalid bounding box geometry", f"{cm.exception}")

    def test_find_places_by_wkt(self):
        ctx = new_test_service_context()
        places = find_places(ctx, "all", geom_wkt="POLYGON ((2 49, 2 55, -1 55, -1 49, 2 49))")
        self._assertPlaceGroup(places, 2, {'0', '3'})

        with self.assertRaises(ServiceBadRequestError) as cm:
            find_places(ctx, "all", geom_wkt="POLYGLON ((2 49, 2 55, -1 55, -1 49, 2 49))")
        self.assertEqual("HTTP 400: Received invalid geometry WKT", f"{cm.exception}")

    def test_find_places_by_geojson(self):
        ctx = new_test_service_context()

        geojson_obj = {'type': 'Polygon',
                       'coordinates': (((2.0, 49.0), (2.0, 55.0), (-1.0, 55.0), (-1.0, 49.0), (2.0, 49.0)),)}
        places = find_places(ctx, "all", geojson_obj=geojson_obj)
        self._assertPlaceGroup(places, 2, {'0', '3'})

        geojson_obj = {'type': 'Feature', 'properties': {}, 'geometry': geojson_obj}
        places = find_places(ctx, "all", geojson_obj=geojson_obj)
        self._assertPlaceGroup(places, 2, {'0', '3'})

        geojson_obj = {'type': 'FeatureCollection', 'places': [geojson_obj]}
        places = find_places(ctx, "all", geojson_obj=geojson_obj)
        self._assertPlaceGroup(places, 2, {'0', '3'})

        with self.assertRaises(ServiceBadRequestError) as cm:
            geojson_obj = {'type': 'FeatureCollection', 'places': []}
            find_places(ctx, "all", geojson_obj=geojson_obj)
        self.assertEqual("HTTP 400: Received invalid GeoJSON object", f"{cm.exception}")

    def test_find_dataset_features(self):
        ctx = new_test_service_context()
        places = find_dataset_places(ctx, "all", "demo")
        self._assertPlaceGroup(places, 3, {'0', '1', '2'})

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
