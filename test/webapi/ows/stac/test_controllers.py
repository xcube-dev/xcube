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
import datetime
import json
import unittest
from pathlib import Path
import functools

import xarray as xr
import xcube
from xcube.core.gridmapping import GridMapping
from xcube.core.new import new_cube
from xcube.server.api import ApiError
from xcube.webapi.ows.stac.config import (
    DEFAULT_CATALOG_DESCRIPTION,
    DEFAULT_FEATURE_ID,
)
from xcube.webapi.ows.stac.config import DEFAULT_CATALOG_ID
from xcube.webapi.ows.stac.config import DEFAULT_CATALOG_TITLE
from xcube.webapi.ows.stac.config import DEFAULT_COLLECTION_DESCRIPTION
from xcube.webapi.ows.stac.config import DEFAULT_COLLECTION_ID
from xcube.webapi.ows.stac.config import DEFAULT_COLLECTION_TITLE
from xcube.webapi.ows.stac.controllers import (
    STAC_VERSION,
    get_collection_queryables,
    get_datacube_dimensions,
    get_single_collection_items,
)
from xcube.webapi.ows.stac.controllers import get_collection
from xcube.webapi.ows.stac.controllers import get_collection_item
from xcube.webapi.ows.stac.controllers import get_datasets_collection_items
from xcube.webapi.ows.stac.controllers import get_collections
from xcube.webapi.ows.stac.controllers import get_conformance
from xcube.webapi.ows.stac.controllers import get_root
from .test_context import get_stac_ctx

PATH_PREFIX = '/ogc'

BASE_URL = "http://localhost:8080"

_OGC_PREFIX = 'http://www.opengis.net/spec/ogcapi-features-1/1.0/conf'
_STAC_PREFIX = 'https://api.stacspec.org'

EXPECTED_CONFORMANCE = {
    'https://api.geodatacube.example/1.0.0-beta',
    'https://api.stacspec.org/v1.0.0/core',
    'https://api.stacspec.org/v1.0.0/collections',
    'https://api.stacspec.org/v1.0.0/ogcapi-features',
    'http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/core',
    'http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/json',
    'http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/oas30',
    'http://www.opengis.net/spec/ogcapi-common-2/1.0/conf/collections',
    'http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/core',
    'http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/oas30',
    'http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/html',
    'http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/geojson',
    'http://www.opengis.net/spec/ogcapi-coverages-1/1.0/conf/geodata-coverage',
    'http://www.opengis.net/spec/ogcapi-coverages-1/1.0/conf/cisjson',
    'http://www.opengis.net/spec/ogcapi-coverages-1/1.0/conf/coverage-subset',
    'http://www.opengis.net/spec/ogcapi-coverages-1/1.0/conf/oas30',
}

EXPECTED_ENDPOINTS = functools.reduce(
    lambda endpoint_list, ep: endpoint_list
    + [{'methods': ep[0], 'path': ep[1] + suffix} for suffix in ('/', '')],
    [
        (['get'], '/collections/{collectionId}/coverage'),
        (['get'], '/collections/{collectionId}/coverage/domainset'),
        (['get'], '/collections/{collectionId}/coverage/rangetype'),
        (['get'], '/collections/{collectionId}/coverage/metadata'),
        (['get'], '/collections/{collectionId}/coverage/rangeset'),
        (['get'], ''),
        (['get'], '/conformance'),
        (['get'], '/collections'),
        (['get'], '/collections/{collectionId}'),
        (['get'], '/collections/{collectionId}/items'),
        (['get'], '/collections/{collectionId}/items/{featureId}'),
        (['get', 'post'], '/search'),
        (['get'], '/collections/{collectionId}/queryables'),
    ],
    [],
)

EXPECTED_DATASETS_COLLECTION = {
    'description': DEFAULT_COLLECTION_DESCRIPTION,
    'extent': {
        'spatial': {'bbox': [[-180.0, -90.0, 180.0, 90.0]]},
        'temporal': {'interval': [['2000-01-01T00:00:00Z', None]]},
    },
    'id': DEFAULT_COLLECTION_ID,
    'keywords': [],
    'license': 'proprietary',
    'links': [
        {
            'href': f'{BASE_URL}{PATH_PREFIX}',
            'rel': 'root',
            'title': 'root of the OGC API and STAC catalog',
            'type': 'application/json',
        },
        {
            'href': f'{BASE_URL}{PATH_PREFIX}/collections/'
            f'{DEFAULT_COLLECTION_ID}',
            'rel': 'self',
            'title': 'this collection',
            'type': 'application/json',
        },
        {
            'href': f'{BASE_URL}{PATH_PREFIX}/collections',
            'rel': 'parent',
            'title': 'collections list',
        },
        {
            'href': f'{BASE_URL}{PATH_PREFIX}/collections/'
            f'{DEFAULT_COLLECTION_ID}/items',
            'rel': 'items',
            'title': 'feature collection of data cube items',
        },
    ]
    + [
        {
            'href': f'http://localhost:8080/ogc/collections/datacubes/items/{dsid}',
            'rel': 'item',
            'title': f'Feature for the dataset "{dsid}"',
            'type': 'application/geo+json',
        }
        for dsid in ['demo', 'demo-1w']
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
    @functools.lru_cache
    def read_json(self, filename):
        with open(Path(__file__).parent / filename, mode='r') as fp:
            content = json.load(fp)
        return content

    # Commented out to keep coverage checkers happy.
    # @staticmethod
    # def write_json(filename, content):
    #     """Convenience function for updating saved expected JSON
    #
    #     Not used during an ordinary test run.
    #     """
    #     with open(Path(__file__).parent / filename, mode='w') as fp:
    #         json.dump(content, fp, indent=2)

    def test_get_datasets_collection_item(self):
        self.maxDiff = None
        result = get_collection_item(
            get_stac_ctx().datasets_ctx,
            BASE_URL,
            DEFAULT_COLLECTION_ID,
            "demo-1w",
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(self.read_json('stac-datacubes-item.json'), result)

    def test_get_single_collection_item(self):
        self.maxDiff = None
        result = get_collection_item(
            get_stac_ctx().datasets_ctx,
            BASE_URL,
            'demo-1w',
            DEFAULT_FEATURE_ID,
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(self.read_json('stac-single-item.json'), result)

    def test_get_collection_item_nonexistent_feature(self):
        self.assertRaises(
            ApiError.NotFound,
            get_collection_item,
            get_stac_ctx().datasets_ctx,
            BASE_URL,
            'demo-1w',
            'this-feature-does-not-exist',
        )

    def test_get_datasets_collection_items(self):
        result = get_datasets_collection_items(
            get_stac_ctx().datasets_ctx, BASE_URL, DEFAULT_COLLECTION_ID
        )
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

    def test_get_single_collection_items(self):
        result = get_single_collection_items(
            get_stac_ctx().datasets_ctx, BASE_URL, "demo-1w"
        )
        expected_item = self.read_json('stac-single-item.json')
        self.assertEqual('FeatureCollection', result['type'])
        self.assertEqual(
            {'root', 'self'}, set([link['rel'] for link in result['links']])
        )
        self.assertLess(
            datetime.datetime.now().astimezone()
            - datetime.datetime.fromisoformat(result['timeStamp']),
            datetime.timedelta(seconds=10),
        )
        for key in 'bbox', 'geometry', 'properties', 'assets':
            self.assertEqual(expected_item[key], result['features'][0][key])

    def test_get_datasets_collection(self):
        result = get_collection(
            get_stac_ctx().datasets_ctx, BASE_URL, DEFAULT_COLLECTION_ID
        )
        self.assertEqual(EXPECTED_DATASETS_COLLECTION, result)

    def test_get_single_collection(self):
        result = get_collection(get_stac_ctx().datasets_ctx, BASE_URL, 'demo')
        self.assertEqual(self.read_json('demo-collection.json'), result)

    def test_get_collections(self):
        result = get_collections(get_stac_ctx().datasets_ctx, BASE_URL)
        self.assertEqual(
            [
                {
                    'href': f'{BASE_URL}{PATH_PREFIX}',
                    'rel': 'root',
                    'title': 'root of the OGC API and STAC catalog',
                    'type': 'application/json',
                },
                {
                    'href': f'{BASE_URL}{PATH_PREFIX}/collections',
                    'rel': 'self',
                    'type': 'application/json',
                },
                {'href': f'{BASE_URL}{PATH_PREFIX}', 'rel': 'parent'},
            ],
            result['links'],
        )
        self.assertEqual(
            EXPECTED_DATASETS_COLLECTION, result['collections'][0]
        )
        self.assertEqual(
            ['datacubes', 'demo', 'demo-1w'],
            [collection['id'] for collection in result['collections']],
        )

    def test_get_conformance(self):
        result = get_conformance()
        self.assertEqual(EXPECTED_CONFORMANCE, set(result.get('conformsTo')))

    def test_get_root(self):
        result = get_root(get_stac_ctx().datasets_ctx, BASE_URL)
        # Handle conformance separately, since we don't care about the order.
        conformance = result.pop('conformsTo')
        self.assertEqual(EXPECTED_CONFORMANCE, set(conformance))
        self.assertEqual(
            {
                'description': DEFAULT_CATALOG_DESCRIPTION,
                'id': DEFAULT_CATALOG_ID,
                'links': [
                    {
                        'href': f'{BASE_URL}{PATH_PREFIX}',
                        'rel': 'root',
                        'title': 'root of the OGC API and STAC catalog',
                        'type': 'application/json',
                    },
                    {
                        'href': f'{BASE_URL}{PATH_PREFIX}',
                        'rel': 'self',
                        'title': 'this document',
                        'type': 'application/json',
                    },
                    {
                        'href': f'{BASE_URL}/openapi.json',
                        'rel': 'service-desc',
                        'title': 'the API definition',
                        'type': 'application/vnd.oai.openapi+json;version=3.0',
                    },
                    {
                        'href': f'{BASE_URL}/openapi.html',
                        'rel': 'service-doc',
                        'title': 'the API documentation',
                        'type': 'text/html',
                    },
                    {
                        'href': f'{BASE_URL}{PATH_PREFIX}/conformance',
                        'rel': 'conformance',
                        'title': 'OGC API conformance classes'
                        ' implemented by this server',
                        'type': 'application/json',
                    },
                    {
                        'href': f'{BASE_URL}{PATH_PREFIX}/collections',
                        'rel': 'data',
                        'title': 'Information about the feature collections',
                        'type': 'application/json',
                    },
                    {
                        'href': f'{BASE_URL}{PATH_PREFIX}/search',
                        'rel': 'search',
                        'title': 'Search across feature collections',
                        'type': 'application/json',
                    },
                    {
                        'href': f'{BASE_URL}{PATH_PREFIX}/collections/'
                        f'{DEFAULT_COLLECTION_ID}',
                        'rel': 'child',
                        'title': DEFAULT_COLLECTION_DESCRIPTION,
                        'type': 'application/json',
                    },
                ],
                'stac_version': STAC_VERSION,
                'title': DEFAULT_CATALOG_TITLE,
                'type': 'Catalog',
                'api_version': '1.0.0',
                'backend_version': xcube.__version__,
                'gdc_version': '1.0.0-beta',
                'endpoints': EXPECTED_ENDPOINTS,
            },
            result,
        )

    def test_get_collection_queryables(self):
        result = get_collection_queryables(
            get_stac_ctx().datasets_ctx, DEFAULT_COLLECTION_ID
        )
        self.assertEqual(
            {
                'additionalProperties': False,
                'properties': {},
                'title': 'datacubes',
                'type': 'object',
            },
            result,
        )

    def test_get_datacube_dimensions(self):
        dim_name = 'a_new_dimension'
        cube = new_cube(variables={'v': 0}).expand_dims({dim_name: 1})
        cube[dim_name] = xr.DataArray([0])
        dims = get_datacube_dimensions(cube, GridMapping.from_dataset(cube))

        expected = {
            'lon': {
                'type': 'spatial',
                'axis': 'x',
                'description': 'longitude',
                'unit': 'degrees_east',
                'extent': [-180, 180],
                'step': 1,
                'reference_system': 'EPSG:4326',
            },
            'lat': {
                'type': 'spatial',
                'axis': 'y',
                'description': 'latitude',
                'unit': 'degrees_north',
                'extent': [-90, 90],
                'step': 1,
                'reference_system': 'EPSG:4326',
            },
            'time': {
                'type': 'temporal',
                'values': [
                    '2010-01-01T12:00:00Z',
                    '2010-01-02T12:00:00Z',
                    '2010-01-03T12:00:00Z',
                    '2010-01-04T12:00:00Z',
                    '2010-01-05T12:00:00Z',
                ],
            },
            dim_name: {'type': 'unknown', 'range': [0, 0], 'values': [0]},
        }
        self.assertEqual(expected, dims)
