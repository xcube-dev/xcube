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


import unittest

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

EXPECTED_COLLECTION = {
    'id': DEFAULT_COLLECTION_ID,
    'title': DEFAULT_COLLECTION_TITLE,
    'description': DEFAULT_COLLECTION_DESCRIPTION,
    'stac_version': STAC_VERSION,
    'stac_extensions': ['xcube'],
    'summaries': {},
    'extent': {},
    'keywords': [],
    'license': 'proprietary',
    'links': [
        {
            'href': f'{BASE_URL}/catalog/collections/{DEFAULT_COLLECTION_ID}',
            'rel': 'self'
        },
        {
            'href': f'{BASE_URL}/catalog/collections',
            'rel': 'root'
        }
    ],
    'providers': [],
}


class StacControllersTest(unittest.TestCase):
    def test_get_collection_item(self):
        result = get_collection_item(get_stac_ctx().datasets_ctx, BASE_URL,
                                     DEFAULT_COLLECTION_ID, "demo-1w")
        self.assertIsInstance(result, dict)
        # TODO (forman): add more assertions

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
        self.assertEqual({'collections': [EXPECTED_COLLECTION]}, result)

    def test_get_conformance(self):
        result = get_conformance(get_stac_ctx().datasets_ctx)
        prefix = 'http://www.opengis.net/spec/ogcapi-features-1/1.0/conf'
        self.assertEqual(
            {
                'conformsTo': [
                    f'{prefix}/core',
                    f'{prefix}/oas30',
                    f'{prefix}/html',
                    f'{prefix}/geojson'
                ]
            },
            result
        )

    def test_get_root(self):
        result = get_root(get_stac_ctx().datasets_ctx, BASE_URL)
        self.assertEqual(
            {
                'stac_version': STAC_VERSION,
                'id': DEFAULT_CATALOG_ID,
                'title': DEFAULT_CATALOG_TITLE,
                'description': DEFAULT_CATALOG_DESCRIPTION,
                'links': [
                    {'rel': 'self',
                     'href': f'{BASE_URL}/catalog',
                     'type': 'application/json',
                     'title': 'this document'},
                    {'rel': 'service-desc',
                     'href': f'{BASE_URL}/openapi.json',
                     'type': 'application/vnd.oai.openapi+json;version=3.0',
                     'title': 'the API definition'},
                    {'rel': 'service-doc',
                     'href': f'{BASE_URL}/openapi.html',
                     'type': 'text/html',
                     'title': 'the API documentation'},
                    {'rel': 'conformance',
                     'href': f'{BASE_URL}/catalog/conformance',
                     'type': 'application/json',
                     'title': 'OGC API conformance classes implemented'
                              ' by this server'},
                    {'rel': 'data',
                     'href': f'{BASE_URL}/catalog/collections',
                     'type': 'application/json',
                     'title': 'Information about the feature collections'},
                    {'rel': 'search',
                     'href': f'{BASE_URL}/catalog/search',
                     'type': 'application/json',
                     'title': 'Search across feature collections'}
                ]
            },
            result
        )
