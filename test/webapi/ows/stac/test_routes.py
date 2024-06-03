# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Any
from collections.abc import Mapping

from xcube.webapi.ows.stac.config import DEFAULT_COLLECTION_ID
from ...helpers import RoutesTestCase, get_res_test_dir

from xcube.webapi.ows.stac.routes import PATH_PREFIX


class StacRoutesTest(RoutesTestCase):
    """STAC endpoints smoke tests"""

    def test_fetch_catalog(self):
        response = self.fetch(PATH_PREFIX + "")
        self.assertResponseOK(response)

    def test_fetch_catalog_conformance(self):
        response = self.fetch(PATH_PREFIX + "/conformance")
        self.assertResponseOK(response)

    def test_fetch_catalog_collections(self):
        response = self.fetch(PATH_PREFIX + "/collections")
        self.assertResponseOK(response)

    def test_fetch_catalog_collection(self):
        response = self.fetch(PATH_PREFIX + "/collections/datasets")
        self.assertResourceNotFoundResponse(response)
        response = self.fetch(PATH_PREFIX + "/collections/datacubes")
        self.assertResponseOK(response)

    def test_fetch_catalog_collection_datacubes_items(self):
        response = self.fetch(PATH_PREFIX + "/collections/datacubes/items")
        self.assertResponseOK(response)
        response = self.fetch(
            PATH_PREFIX + "/collections/datacubes/items" "?limit=1&cursor=1"
        )
        self.assertResponseOK(response)
        response = self.fetch(PATH_PREFIX + "/collections/datasets")
        self.assertResourceNotFoundResponse(response)

    def test_fetch_catalog_collection_single_items(self):
        response = self.fetch(PATH_PREFIX + "/collections/demo/items")
        self.assertResponseOK(response)

    def test_fetch_catalog_collection_item(self):
        response = self.fetch(PATH_PREFIX + "/collections/datacubes/items/demo")
        self.assertResponseOK(response)
        response = self.fetch(PATH_PREFIX + "/collections/datacubes/items/demox")
        self.assertResourceNotFoundResponse(response)
        response = self.fetch(PATH_PREFIX + "/collections/datasets/items/demo")
        self.assertResourceNotFoundResponse(response)

    def test_fetch_catalog_search_by_kw(self):
        response = self.fetch(PATH_PREFIX + "/search", method="GET")
        self.assertResponseOK(response)

    def test_fetch_catalog_search_by_json(self):
        response = self.fetch(PATH_PREFIX + "/search", method="POST")
        self.assertResponseOK(response)

    def test_fetch_collection_queryables(self):
        response = self.fetch(
            f"{PATH_PREFIX}/collections/{DEFAULT_COLLECTION_ID}/queryables",
            method="GET",
        )
        self.assertResponseOK(response)

    def test_fetch_collection_schema(self):
        response = self.fetch(f"{PATH_PREFIX}/collections/demo/schema", method="GET")
        self.assertResponseOK(response)


class StacRoutesTestCog(RoutesTestCase):
    def get_config(self) -> Mapping[str, Any]:
        return {
            "Datasets": [
                {
                    "Identifier": "demo",
                    "Title": "xcube-server COG sample",
                    "Path": f"{get_res_test_dir()}/../../../"
                    f"examples/serve/demo/sample-cog.tif",
                }
            ]
        }

    def test_fetch_catalog_collection_items(self):
        response = self.fetch(PATH_PREFIX + "/collections/datacubes/items")
        self.assertResponseOK(response)
