# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest
from unittest.mock import MagicMock

import fiona
import shapely
from shapely.geometry import mapping

from test.webapi.helpers import get_api_ctx
from xcube.server.api import Context
from xcube.webapi.places.context import PlacesContext


def get_places_ctx() -> PlacesContext:
    return get_api_ctx("places", PlacesContext, "config.yml")


class PlacesContextTest(unittest.TestCase):
    def test_ctx_ok(self):
        ctx = get_places_ctx()
        self.assertIsInstance(ctx.server_ctx, Context)
        self.assertIsInstance(ctx.auth_ctx, Context)

    def test_find_dataset_features_with_additional_place_group(self):
        ctx = get_places_ctx()
        place_group = dict(
            type="FeatureCollection",
            features=None,
            id="test",
            title="title_0",
            propertyMapping=None,
            sourcePaths="None",
        )
        ctx.add_place_group(place_group, ["dataset_0"])
        place_groups = ctx.load_place_groups([], "", False, False, ["dataset_0"])
        self.assertEqual(1, len(place_groups))
        self.assertEqual("title_0", place_groups[0]["title"])

    def test_reprojection(self):
        c = MagicMock(crs=fiona.crs.CRS.from_epsg(3287))
        shapely_geometry = shapely.wkt.loads(
            "POLYGON((8 55, 10 55, 10 57, 8 57, 8 55))"
        )

        feature = {"geometry": mapping(shapely_geometry)}

        c.__iter__.return_value = [feature]
        fc = PlacesContext._to_geo_interface(c)
        transformed_coords = list(fc)[0]["geometry"]["coordinates"][0]

        self.assertEqual(-141.72410717292482, transformed_coords[0][0])
        self.assertEqual(-89.9994987734214, transformed_coords[0][1])
        self.assertEqual(-141.72410717292482, transformed_coords[4][0])
        self.assertEqual(-89.9994987734214, transformed_coords[4][1])

        shapely_geometry = shapely.wkt.loads(
            "POLYGON((8 55, 10 55, 10 57, 8 57, 8 55))"
        )

        feature = {"geometry": mapping(shapely_geometry)}

        c = MagicMock(crs=fiona.crs.CRS.from_epsg(4326))
        c.__iter__.return_value = [feature]
        fc = PlacesContext._to_geo_interface(c)
        transformed_coords = list(fc)[0]["geometry"]["coordinates"][0]

        self.assertEqual(8, transformed_coords[0][0])
        self.assertEqual(55, transformed_coords[0][1])
        self.assertEqual(8, transformed_coords[4][0])
        self.assertEqual(55, transformed_coords[4][1])
