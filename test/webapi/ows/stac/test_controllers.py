# The MIT License (MIT)
# Copyright (c) 2023 by the xcube team and contributors
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
import os.path
import unittest
from pathlib import Path

from xcube.webapi.ows.stac.config import DEFAULT_CATALOG_DESCRIPTION
from xcube.webapi.ows.stac.config import DEFAULT_CATALOG_ID
from xcube.webapi.ows.stac.config import DEFAULT_CATALOG_TITLE
from xcube.webapi.ows.stac.config import DEFAULT_COLLECTION_DESCRIPTION
from xcube.webapi.ows.stac.config import DEFAULT_COLLECTION_ID
from xcube.webapi.ows.stac.config import DEFAULT_COLLECTION_TITLE
from xcube.webapi.ows.stac.controllers import STAC_VERSION
from xcube.webapi.ows.stac.controllers import get_collection
from xcube.webapi.ows.stac.controllers import get_collection_item
from xcube.webapi.ows.stac.controllers import get_collection_items
from xcube.webapi.ows.stac.controllers import get_collections
from xcube.webapi.ows.stac.controllers import get_conformance
from xcube.webapi.ows.stac.controllers import get_root
from .test_context import get_stac_ctx

BASE_URL = "http://localhost:8080"

_OGC_PREFIX = 'http://www.opengis.net/spec/ogcapi-features-1/1.0/conf'
_STAC_PREFIX = 'https://api.stacspec.org'

EXPECTED_CONFORMANCE = [
    f'{_STAC_PREFIX}/v1.0.0-rc.2/core',
    f'{_STAC_PREFIX}/v1.0.0-rc.2/ogcapi-features',
    f'{_STAC_PREFIX}/v1.0.0-rc.1/collections',
    f'{_OGC_PREFIX}/core',
    f'{_OGC_PREFIX}/oas30',
    f'{_OGC_PREFIX}/html',
    f'{_OGC_PREFIX}/geojson'
]

EXPECTED_COLLECTION = {
    'description': DEFAULT_COLLECTION_DESCRIPTION,
    'extent': {'spatial': {'bbox': [[-180.0, -90.0, 180.0, 90.0]]},
               'temporal': {'interval': [['2000-01-01T00:00:00Z', None]]}},
    'id': DEFAULT_COLLECTION_ID,
    'keywords': [],
    'license': 'proprietary',
    'links': [
        {'href': f'{BASE_URL}/catalog',
         'rel': 'root',
         'title': 'root of the STAC catalog',
         'type': 'application/json'},
        {'href': f'{BASE_URL}/catalog/collections/{DEFAULT_COLLECTION_ID}',
         'rel': 'self',
         'title': 'this collection',
         'type': 'application/json'},
        {'href': f'{BASE_URL}/catalog/collections',
         'rel': 'parent',
         'title': 'collections list'},
        {
            'href': f'{BASE_URL}/catalog/collections/{DEFAULT_COLLECTION_ID}/items',
            'rel': 'items',
            'title': 'feature collection of data cube items'}
    ],
    'providers': [],
    'stac_version': STAC_VERSION,
    'stac_extensions': [
        'https://stac-extensions.github.io/datacube/v2.1.0/schema.json'
    ],
    'summaries': {},
    'title': DEFAULT_COLLECTION_TITLE,
    'type': 'Collection',
}


class StacControllersTest(unittest.TestCase):
    def test_get_collection_item(self):
        self.maxDiff = None
        result = get_collection_item(get_stac_ctx().datasets_ctx, BASE_URL,
                                     DEFAULT_COLLECTION_ID, "demo-1w")
        self.assertIsInstance(result, dict)
        path = Path(__file__).parent / "stac-item.json"
        # with open(path, mode="w") as fp:
        #     json.dump(result, fp, indent=2)
        with open(path, mode="r") as fp:
            expected_result = json.load(fp)
        self.assertEqual(expected_result, result)

    def test_get_collection_items(self):
        result = get_collection_items(get_stac_ctx().datasets_ctx, BASE_URL,
                                      DEFAULT_COLLECTION_ID)
        self.assertIsInstance(result, dict)
        self.assertIn('features', result)

        features = result['features']
        self.assertIsInstance(features, list)
        self.assertEqual(2, len(features))

        for feature in features:
            self.assertIsInstance(feature, dict)
            self.assertEqual('Feature', feature.get('type'))
            self.assertEqual(DEFAULT_COLLECTION_ID, feature.get('collection'))
            self.assertIsInstance(feature.get('bbox'), list)
            self.assertIsInstance(feature.get('properties'), dict)
            self.assertIsInstance(feature.get('geometry'), dict)
            self.assertIsInstance(feature.get('assets'), dict)

            self.assertIsInstance(feature.get('id'), str)
            self.assertIn(feature['id'], {'demo', 'demo-1w'})
            # TODO (forman): add more assertions
            # import pprint
            # pprint.pprint(feature)

    def test_get_collection(self):
        result = get_collection(get_stac_ctx().datasets_ctx, BASE_URL,
                                DEFAULT_COLLECTION_ID)
        self.assertEqual(EXPECTED_COLLECTION, result)

    def test_get_collections(self):
        result = get_collections(get_stac_ctx().datasets_ctx, BASE_URL)
        self.assertEqual(
            {'collections': [EXPECTED_COLLECTION],
             'links': [{'href': f'{BASE_URL}/catalog',
                        'rel': 'root',
                        'title': 'root of the STAC catalog',
                        'type': 'application/json'},
                       {'href': f'{BASE_URL}/catalog/collections',
                        'rel': 'self',
                        'type': 'application/json'},
                       {'href': f'{BASE_URL}/catalog', 'rel': 'parent'}]},
            result
        )

    def test_get_conformance(self):
        result = get_conformance(get_stac_ctx().datasets_ctx)
        self.assertEqual(
            {
                'conformsTo': EXPECTED_CONFORMANCE
            },
            result
        )

    def test_get_root(self):
        result = get_root(get_stac_ctx().datasets_ctx, BASE_URL)
        self.assertEqual(
            {
                'conformsTo': EXPECTED_CONFORMANCE,
                'description': DEFAULT_CATALOG_DESCRIPTION,
                'id': DEFAULT_CATALOG_ID,
                'links': [
                    {'href': f'{BASE_URL}/catalog',
                     'rel': 'root',
                     'title': 'root of the STAC catalog',
                     'type': 'application/json'},
                    {'href': f'{BASE_URL}/catalog',
                     'rel': 'self',
                     'title': 'this document',
                     'type': 'application/json'},
                    {'href': f'{BASE_URL}/openapi.json',
                     'rel': 'service-desc',
                     'title': 'the API definition',
                     'type': 'application/vnd.oai.openapi+json;version=3.0'},
                    {'href': f'{BASE_URL}/openapi.html',
                     'rel': 'service-doc',
                     'title': 'the API documentation',
                     'type': 'text/html'},
                    {'href': f'{BASE_URL}/catalog/conformance',
                     'rel': 'conformance',
                     'title': 'OGC API conformance classes'
                              ' implemented by this server',
                     'type': 'application/json'},
                    {'href': f'{BASE_URL}/catalog/collections',
                     'rel': 'data',
                     'title': 'Information about the feature collections',
                     'type': 'application/json'},
                    {'href': f'{BASE_URL}/catalog/search',
                     'rel': 'search',
                     'title': 'Search across feature collections',
                     'type': 'application/json'},
                    {
                        'href': f'{BASE_URL}/catalog/collections/'
                                f'{DEFAULT_COLLECTION_ID}',
                        'rel': 'child',
                        'title': DEFAULT_COLLECTION_DESCRIPTION,
                        'type': 'application/json'}
                ],
                'stac_version': STAC_VERSION,
                'title': DEFAULT_CATALOG_TITLE,
                'type': 'Catalog'
            },
            result
        )
