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
from typing import Hashable

import pyproj
import xarray as xr
from xcube.core.gridmapping import CRS_CRS84

from ...datasets.context import DatasetsContext
from .config import DEFAULT_CATALOG_ID
from .config import DEFAULT_CATALOG_TITLE
from .config import DEFAULT_CATALOG_DESCRIPTION

STAC_VERSION = '0.9.0'

DATA_CUBES_COLLECTION_ID = "datacubes"


# noinspection PyUnusedLocal
def get_root(ctx: DatasetsContext, base_url: str):
    stac_config = ctx.config.get("STAC", {})
    return {
        "stac_version": STAC_VERSION,
        "id": stac_config.get("Identifier",
                              DEFAULT_CATALOG_ID),
        "title": stac_config.get("Title",
                                 DEFAULT_CATALOG_TITLE),
        "description": stac_config.get("Description",
                                       DEFAULT_CATALOG_DESCRIPTION),
        "links": [
            {
                "rel": "self",
                "href": f'{base_url}/catalog',
                "type": "application/json",
                "title": "this document"
            },
            {
                "rel": "service-desc",
                "href": f'{base_url}/openapi.json',
                "type": "application/vnd.oai.openapi+json;version=3.0",
                "title": "the API definition"
            },
            {
                "rel": "service-doc",
                "href": f'{base_url}/openapi.html',
                "type": "text/html",
                "title": "the API documentation"
            },
            {
                "rel": "conformance",
                "href": f'{base_url}/catalog/conformance',
                "type": "application/json",
                "title": "OGC API conformance classes"
                         " implemented by this server"
            },
            {
                "rel": "data",
                "href": f'{base_url}/catalog/collections',
                "type": "application/json",
                "title": "Information about the feature collections"
            },
            {
                "rel": "search",
                "href": f'{base_url}/catalog/search',
                "type": "application/json",
                "title": "Search across feature collections"
            }
        ],
    }


# noinspection PyUnusedLocal
def get_conformance(ctx: DatasetsContext):
    return {
        "conformsTo": [
            # TODO: fix this list
            "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/core",
            "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/oas30",
            "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/html",
            "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/geojson"
        ]
    }


def get_collections(ctx: DatasetsContext, base_url: str):
    return {
        "collections": [
            _get_datacubes_collection(ctx, base_url)
        ]
    }


def get_collection(ctx: DatasetsContext,
                   base_url: str,
                   collection_id: str):
    if collection_id != DATA_CUBES_COLLECTION_ID:
        raise CollectionNotFoundException()
    return _get_datacubes_collection(ctx, base_url, full=True)


def get_collection_items(ctx: DatasetsContext,
                         base_url: str,
                         collection_id: str):
    if collection_id != DATA_CUBES_COLLECTION_ID:
        raise CollectionNotFoundException()

    features = []
    for dataset_config in ctx.get_dataset_configs():
        dataset_id = dataset_config["Identifier"]
        feature = _get_dataset_feature(ctx,
                                       base_url,
                                       dataset_id,
                                       full=False)
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features,
        "timeStamp": _utc_now(),
        "numberMatched": len(features),
        "numberReturned": len(features),
    }


def get_collection_item(ctx: DatasetsContext,
                        base_url: str,
                        collection_id: str,
                        feature_id: str):
    if collection_id != DATA_CUBES_COLLECTION_ID:
        raise CollectionNotFoundException()
    return _get_dataset_feature(ctx, base_url, feature_id, full=True)


# noinspection PyUnusedLocal
def search(ctx: DatasetsContext, base_url: str):
    # TODO: implement me!
    return {}


# noinspection PyUnusedLocal
def _get_datacubes_collection(ctx: DatasetsContext,
                              base_url: str,
                              full: bool = False):
    return {
        "stac_version": STAC_VERSION,
        "stac_extensions": ["xcube"],
        "id": DATA_CUBES_COLLECTION_ID,
        "title": "Data cubes",
        "description": "",
        "license": "proprietary",
        "keywords": [],
        "providers": [],
        "extent": {},
        "summaries": {},
        "links": [
            {
                "rel": "self",
                "href": f"{base_url}/catalog/collections/"
                        f"{DATA_CUBES_COLLECTION_ID}"
            },
            {
                "rel": "root",
                "href": f"{base_url}/catalog/collections"
            },
            # {
            #     "rel": "license",
            #     "href": ctx.get_url("TODO"),
            #     "title": "TODO"
            # }
        ]
    }


# noinspection PyUnusedLocal
def _get_dataset_feature(ctx: DatasetsContext,
                         base_url: str,
                         dataset_id: str,
                         full: bool = False):
    ml_dataset = ctx.get_ml_dataset(dataset_id)
    dataset = ml_dataset.base_dataset

    variables = [get_variable_asset(var_name, var)
                 for var_name, var in dataset.data_vars.items()]
    coordinates = [get_variable_asset(var_name, var)
                   for var_name, var in dataset.coords.items()]

    first_var_name = next(iter(variables))["name"]
    first_var = dataset[first_var_name]
    first_var_extra_dims = first_var.dims[0:-2]

    thumbnail_query = ''
    if first_var_extra_dims:
        thumbnail_query_params = []
        for dim in first_var_extra_dims:
            val = 0
            if dim in dataset:
                coord = dataset[dim]
                if coord.ndim == 1 and coord.size > 0:
                    val = coord[0]
            thumbnail_query_params.append(f'{dim}={val}')
        thumbnail_query = '?' + '&'.join(thumbnail_query_params)

    tiles_query = ''
    if first_var_extra_dims:
        tiles_query = '?' + '&'.join(
            ['%s=<%s>' % (d, d) for d in first_var_extra_dims]
        )

    t = pyproj.Transformer.from_crs(
        ml_dataset.grid_mapping.crs,
        CRS_CRS84,
        always_xy=True
    )
    bbox = ml_dataset.grid_mapping.xy_bbox
    (x1, x2), (y1, y2) = t.transform((bbox[0], bbox[2]), (bbox[1], bbox[3]))

    # TODO: Prefer original storage location.
    #       The "s3" operation is default.
    default_storage_url = f"{base_url}/s3/datasets"

    return {
        "stac_version": STAC_VERSION,
        "stac_extensions": ["xcube"],
        "type": "Feature",
        "id": dataset_id,
        "bbox": [x1, y1, x2, y2],
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]],
            ],
        },
        "properties": {
            "xcube:dimensions": dict(dataset.dims),
            "xcube:variables": variables,
            "xcube:coordinates": coordinates,
            "xcube:attributes": dict(dataset.attrs),
        },
        "collection": DATA_CUBES_COLLECTION_ID,
        "links": [
            {
                "rel": "self",
                'href': f'{base_url}/catalog/collections/{dataset_id}'
            }
        ],
        "assets": {
            "analytic": {
                "title": f"{dataset_id} data access",
                "roles": ["data"],
                "type": "application/zarr",
                "href": f"{default_storage_url}/{dataset_id}.zarr",
                "xcube:analytic": {
                    v['name']: {
                        "title": f"{v['name']} data access",
                        "roles": ["data"],
                        "type": "application/zarr",
                        "href": f"{default_storage_url}/"
                                f"{dataset_id}.zarr/{v['name']}"
                    }
                    for v in variables
                }
            },
            "visual": {
                "title": f"{dataset_id} visualisation",
                "roles": ["visual"],
                "type": "image/png",
                "href": (f"{base_url}/tiles/{dataset_id}/<variable>"
                         + "/{z}/{y}/{x}"
                         + tiles_query),
                "xcube:visual": {
                    v['name']: {
                        "title": f"{v['name']} visualisation",
                        "roles": ["visual"],
                        "type": "image/png",
                        "href": (
                                f"{base_url}/tiles/{dataset_id}/{v['name']}"
                                + "/{z}/{y}/{x}"
                                + tiles_query),
                    }
                    for v in variables
                }
            },
            "thumbnail": {
                "title": f"{dataset_id} thumbnail",
                "roles": ["thumbnail"],
                "type": "image/png",
                "href": f"{base_url}/tiles/{dataset_id}/{first_var_name}"
                        f"/0/0/0{thumbnail_query}"
            }
        }
    }


def get_variable_asset(var_name: Hashable, var: xr.DataArray):
    return {
        "name": str(var_name),
        "dtype": str(var.dtype),
        "dims": list(var.dims),
        "chunks": list(var.chunks) if var.chunks else None,
        "shape": list(var.shape),
        "attrs": dict(var.attrs),
        # "encoding": dict(var.encoding),
    }


def _utc_now():
    return datetime \
               .datetime \
               .utcnow() \
               .replace(microsecond=0) \
               .isoformat() + 'Z'


class CollectionNotFoundException(Exception):
    pass


class DimensionNotFoundException(Exception):
    pass
