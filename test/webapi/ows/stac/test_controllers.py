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

from xcube.webapi.ows.stac.controllers import get_collection
from xcube.webapi.ows.stac.controllers import get_collection_item
from xcube.webapi.ows.stac.controllers import get_collection_items
from xcube.webapi.ows.stac.controllers import get_collections
from xcube.webapi.ows.stac.controllers import get_conformance
from xcube.webapi.ows.stac.controllers import get_root

from .test_context import get_stac_ctx

BASE_URL = "http://localhost:8080"


class StacControllersTest(unittest.TestCase):
    def test_get_collection_item(self):
        result = get_collection_item(get_stac_ctx().datasets_ctx, BASE_URL,
                                     "datacubes", "demo-1w")
        self.assertIsInstance(result, dict)
        # TODO (forman): add more assertions

    def test_get_collection_items(self):
        result = get_collection_items(get_stac_ctx().datasets_ctx, BASE_URL,
                                      "datacubes")
        self.assertIsInstance(result, dict)
        self.assertIn('features', result)

        features = result['features']
        self.assertIsInstance(features, list)
        self.assertEqual(2, len(features))

        for feature in features:
            self.assertIsInstance(feature, dict)
            self.assertEqual('Feature', feature.get('type'))
            self.assertEqual('datacubes', feature.get('collection'))
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
                                "datacubes")
        self.assertEqual(
            {
                'id': 'datacubes',
                'title': 'Data cubes',
                'description': '',
                'stac_version': '0.9.0',
                'stac_extensions': ['xcube'],
                'summaries': {},
                'extent': {},
                'keywords': [],
                'license': 'proprietary',
                'links': [
                    {
                        'href': f'{BASE_URL}/catalog/collections'
                                f'/datacubes',
                        'rel': 'self'
                    },
                    {
                        'href': f'{BASE_URL}/catalog/collections',
                        'rel': 'root'
                    }
                ],
                'providers': [],
            },
            result
        )

    def test_get_collections(self):
        result = get_collections(get_stac_ctx().datasets_ctx, BASE_URL)
        self.assertEqual(
            {
                'collections': [
                    {
                        'id': 'datacubes',
                        'title': 'Data cubes',
                        'description': '',
                        'stac_version': '0.9.0',
                        'stac_extensions': ['xcube'],
                        'summaries': {},
                        'extent': {},
                        'keywords': [],
                        'license': 'proprietary',
                        'links': [
                            {
                                'href': f'{BASE_URL}/catalog/collections'
                                        f'/datacubes',
                                'rel': 'self'
                            },
                            {
                                'href': f'{BASE_URL}/catalog/collections',
                                'rel': 'root'
                            }
                        ],
                        'providers': [],
                    }
                ]
            },
            result
        )

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
                'stac_version': '0.9.0', 'id': 'xcube-server',
                'title': 'xcube Server',
                'description': 'Catalog of datasets and places'
                               ' served by xcube.',
                'links': [
                    {'rel': 'self',
                     'href': 'http://localhost:8080/catalog',
                     'type': 'application/json',
                     'title': 'this document'},
                    {'rel': 'service-desc',
                     'href': 'http://localhost:8080/openapi.json',
                     'type': 'application/vnd.oai.openapi+json;version=3.0',
                     'title': 'the API definition'},
                    {'rel': 'service-doc',
                     'href': 'http://localhost:8080/openapi.html',
                     'type': 'text/html',
                     'title': 'the API documentation'},
                    {'rel': 'conformance',
                     'href': 'http://localhost:8080/catalog/conformance',
                     'type': 'application/json',
                     'title': 'OGC API conformance classes implemented'
                              ' by this server'},
                    {'rel': 'data',
                     'href': 'http://localhost:8080/catalog/collections',
                     'type': 'application/json',
                     'title': 'Information about the feature collections'},
                    {'rel': 'search',
                     'href': 'http://localhost:8080/catalog/search',
                     'type': 'application/json',
                     'title': 'Search across feature collections'}
                ]
            },
            result
        )
