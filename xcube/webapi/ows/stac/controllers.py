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
from typing import Hashable, Any, Optional, Dict, List, Mapping
import itertools

import numpy as np
import pyproj
import xarray as xr

import xcube
from xcube.core.gridmapping import CRS_CRS84, GridMapping
from xcube.server.api import ApiError
from xcube.server.api import ServerConfig
from xcube.util.jsonencoder import to_json_value
from xcube.util.jsonschema import JsonObjectSchema, JsonSchema
from .config import DEFAULT_CATALOG_DESCRIPTION, DEFAULT_FEATURE_ID
from .config import DEFAULT_CATALOG_ID
from .config import DEFAULT_CATALOG_TITLE
from .config import DEFAULT_COLLECTION_DESCRIPTION
from .config import DEFAULT_COLLECTION_ID
from .config import DEFAULT_COLLECTION_TITLE
from .config import PATH_PREFIX
from ...datasets.context import DatasetsContext

STAC_VERSION = '1.0.0'
STAC_EXTENSIONS = [
    "https://stac-extensions.github.io/datacube/v2.1.0/schema.json"
]

# Maximum number of values allowed for the "values" field
# of a value of "datacube:dimensions" or "datacube:variables":
_MAX_NUM_VALUES = 1000

_CONFORMANCE = [
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
    'http://www.opengis.net/spec/ogcapi-coverages-1/1.0/conf/oas30'
]


# noinspection PyUnusedLocal
def get_root(ctx: DatasetsContext, base_url: str):
    c_id, c_title, c_description = _get_catalog_metadata(ctx.config)
    endpoint_lists = [a.endpoints() for a in ctx.apis
                      if a.name in {'ows.stac', 'ows.coverages'}]
    return {
        "type": "Catalog",
        "stac_version": STAC_VERSION,
        "conformsTo": _CONFORMANCE,
        "id": c_id,
        "title": c_title,
        "description": c_description,
        "api_version": "1.0.0",
        "backend_version": xcube.__version__,
        "gdc_version": "1.0.0-beta.1",
        "endpoints": list(itertools.chain.from_iterable(endpoint_lists)),
        "links": [
            _root_link(base_url),
            {
                "rel": "self",
                "href": f'{base_url}/ogc',
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
                "href": f'{base_url}{PATH_PREFIX}/conformance',
                "type": "application/json",
                "title": "OGC API conformance classes"
                         " implemented by this server"
            },
            {
                "rel": "data",
                "href": f'{base_url}{PATH_PREFIX}/collections',
                "type": "application/json",
                "title": "Information about the feature collections"
            },
            {
                "rel": "search",
                "href": f'{base_url}{PATH_PREFIX}/search',
                "type": "application/json",
                "title": "Search across feature collections"
            },
            {
                "rel": "child",
                "href": f'{base_url}{PATH_PREFIX}/collections/datacubes',
                "type": "application/json",
                "title": DEFAULT_COLLECTION_DESCRIPTION
            }
        ],
    }


def _root_link(base_url):
    return {
        "rel": "root",
        "href": f'{base_url}/ogc',
        "type": "application/json",
        "title": "root of the OGC API and STAC catalog"
    }


# noinspection PyUnusedLocal
def get_conformance(ctx: DatasetsContext):
    return {"conformsTo": _CONFORMANCE}


def get_collections(ctx: DatasetsContext, base_url: str):
    return {
        "collections": [
            _get_dataset_collection(ctx, base_url, c['Identifier'])
            for c in ctx.get_dataset_configs()
            # _get_datasets_collection(ctx, base_url)
        ],
        "links": [
            _root_link(base_url),
            {
                "rel": "self",
                "type": "application/json",
                "href": f"{base_url}{PATH_PREFIX}/collections"
            },
            {
                "rel": "parent",
                "href": f"{base_url}{PATH_PREFIX}"
            }
        ]
    }


def get_collection(ctx: DatasetsContext,
                   base_url: str,
                   collection_id: str):
    all_datasets_collection_id, _, _ = _get_collection_metadata(ctx.config)
    collection_ids = [c['Identifier'] for c in ctx.get_dataset_configs()]
    if collection_id in collection_ids:
        return _get_dataset_collection(ctx, base_url, collection_id, full=True)
    elif collection_id == all_datasets_collection_id:
        return _get_datasets_collection(ctx, base_url, full=True)
    else:
        raise ApiError.NotFound(f'Collection "{collection_id}" not found')


def get_single_collection_items(
    ctx: DatasetsContext, base_url: str, collection_id: str
):
    feature = _get_dataset_feature(ctx, base_url, collection_id, full=False)
    self_href = f"{base_url}{PATH_PREFIX}/collections/{collection_id}/items"
    return {
        'type': 'FeatureCollection',
        'features': [feature],
        'links': [
            _root_link(base_url),
            {
                'rel': 'self',
                'type': 'application/json',
                'href': self_href
            }
        ],
        'timeStamp': datetime.datetime.now().astimezone().isoformat()
    }


def get_datasets_collection_items(ctx: DatasetsContext,
                                  base_url: str,
                                  collection_id: str,
                                  limit: int = 100,
                                  cursor: int = 0):
    _assert_valid_collection(ctx, collection_id)
    all_configs = ctx.get_dataset_configs()
    configs = all_configs[cursor:(cursor + limit)]
    features = []
    for dataset_config in configs:
        dataset_id = dataset_config["Identifier"]
        feature = _get_dataset_feature(ctx,
                                       base_url,
                                       dataset_id,
                                       full=False)
        features.append(feature)
    self_href = f"{base_url}{PATH_PREFIX}/collections/{collection_id}/items"
    links = [
            _root_link(base_url),
            {
                "rel": "self",
                "type": "application/json",
                "href": self_href
            }
        ]
    if cursor + limit < len(all_configs):
        links.append({
            'rel': 'next',
            'href': self_href + f'?cursor={cursor + limit}&limit={limit}'
        })
    if cursor > 0:
        new_cursor = cursor - limit
        if new_cursor < 0:
            new_cursor = 0
        cursor_param = 'cursor={new_cursor}&' if new_cursor > 0 else ''
        links.append({
            'rel': 'previous',
            'href': self_href + f'?{cursor_param}limit={limit}'
        })
    return {
        "type": "FeatureCollection",
        "features": features,
        "timeStamp": _utc_now(),
        "numberMatched": len(features),
        "numberReturned": len(features),
        "links": links
    }


def get_collection_item(ctx: DatasetsContext,
                        base_url: str,
                        collection_id: str,
                        feature_id: str):
    dataset_ids = {c['Identifier'] for c in ctx.get_dataset_configs()}

    feature_not_found = ApiError.NotFound(
        f'Feature "{feature_id}" not found in collection {collection_id}.'
    )
    if collection_id == DEFAULT_COLLECTION_ID:
        if feature_id in dataset_ids:
            return _get_dataset_feature(ctx, base_url, feature_id, full=True)
        else:
            raise feature_not_found
    elif collection_id in dataset_ids:
        if feature_id == DEFAULT_FEATURE_ID:
            return _get_dataset_feature(ctx, base_url, collection_id, full=True)
        else:
            raise feature_not_found
    else:
        raise ApiError.NotFound(f'Collection "{collection_id}" not found.')


def get_collection_queryables(
    ctx: DatasetsContext, collection_id: str
) -> JsonSchema:
    _assert_valid_collection(ctx, collection_id)
    schema = JsonObjectSchema(
        title=collection_id, properties={}, additional_properties=False
    )
    return schema


# noinspection PyUnusedLocal
def search(ctx: DatasetsContext, base_url: str):
    # TODO: implement me!
    return {}


# noinspection PyUnusedLocal
def _get_datasets_collection(ctx: DatasetsContext,
                             base_url: str,
                             full: bool = False) -> dict:
    c_id, c_title, c_description = _get_collection_metadata(ctx.config)
    return {
        "stac_version": STAC_VERSION,
        "stac_extensions": STAC_EXTENSIONS,
        "id": c_id,
        "type": "Collection",
        "title": c_title,
        "description": c_description,
        "license": "proprietary",
        "keywords": [],
        "providers": [],
        "extent": {
            # TODO Replace these placeholder spatial / temporal extents
            # with extents calculated from the datasets.
            "spatial": {"bbox": [[-180.0, -90.0, 180.0, 90.0]]},
            "temporal": {"interval": [["2000-01-01T00:00:00Z", None]]}
        },
        "summaries": {},
        "links": [
            _root_link(base_url),
            {
                "rel": "self",
                "type": "application/json",
                "href": f"{base_url}{PATH_PREFIX}/collections/{c_id}",
                "title": "this collection"
            },
            {
                "rel": "parent",
                "href": f"{base_url}{PATH_PREFIX}/collections",
                "title": "collections list"
            },
            {
                "rel": "items",
                "href": f"{base_url}{PATH_PREFIX}/collections/{c_id}/items",
                "title": "feature collection of data cube items"
            }
            # {
            #     "rel": "license",
            #     "href": ctx.get_url("TODO"),
            #     "title": "TODO"
            # }
        ]
    }


def _get_dataset_collection(
    ctx: DatasetsContext, base_url: str, dataset_id: str, full: bool = False
) -> dict:
    ml_dataset = ctx.get_ml_dataset(dataset_id)
    dataset = ml_dataset.base_dataset
    feature = _get_dataset_feature(ctx, base_url, dataset_id, full)
    time_properties = _get_time_properties(dataset)
    if {'start_datetime', 'end_datetime'}.issubset(time_properties):
        time_interval = [
            time_properties['start_datetime'],
            time_properties['end_datetime']
        ]
    else:
        time_interval = [
            time_properties['datetime'],
            time_properties['datetime']
        ]
    return {
        'description': dataset_id,
        'extent': {
            'spatial': {
                'bbox': feature['bbox']
            },
            'temporal': {
                'interval': [time_interval]
            }
        },
        'id': dataset_id,
        'keywords': [],
        'license': 'proprietary',
        'links': [
            _root_link(base_url),
            {
                'rel': 'self',
                'type': 'application/json',
                'href': f'{base_url}{PATH_PREFIX}/collections/{dataset_id}',
                'title': 'this collection'
            },
            {
                'rel': 'parent',
                'href': f'{base_url}{PATH_PREFIX}/collections',
                'title': 'collections list'
            },
            {
                'rel': 'items',
                'href': f'{base_url}{PATH_PREFIX}/collections/'
                        f'{dataset_id}/items',
                'title': 'feature collection of data cube items'
            }
        ],
        'providers': [],
        'stac_version': STAC_VERSION,
        "summaries": {},
        'title': dataset_id,
        'type': 'Collection',
    }


# noinspection PyUnusedLocal
def _get_dataset_feature(ctx: DatasetsContext,
                         base_url: str,
                         dataset_id: str,
                         full: bool = False) -> dict:
    collection_id, _, _ = _get_collection_metadata(ctx.config)

    ml_dataset = ctx.get_ml_dataset(dataset_id)
    grid_mapping = ml_dataset.grid_mapping
    dataset = ml_dataset.base_dataset

    xcube_data_vars = _get_xc_variables(dataset.data_vars)
    cube_dimensions = _get_dc_dimensions(dataset, grid_mapping)

    cube_properties = {
        "cube:dimensions": cube_dimensions,
        "cube:variables": (_get_dc_variables(dataset, cube_dimensions)),
        "xcube:dims": to_json_value(dataset.dims),
        "xcube:data_vars": xcube_data_vars,
        "xcube:coords": (_get_xc_variables(dataset.coords)),
        "xcube:attrs": to_json_value(dataset.attrs),
        **(_get_time_properties(dataset)),
    }

    first_var_name = next(iter(xcube_data_vars))["name"]
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
                    val = coord[0].to_numpy()
            thumbnail_query_params.append(f'{dim}={val}')
        thumbnail_query = '?' + '&'.join(thumbnail_query_params)

    tiles_query = ''
    if first_var_extra_dims:
        tiles_query = '?' + '&'.join(
            ['%s=<%s>' % (d, d) for d in first_var_extra_dims]
        )

    def transform_bbox(grid_mapping) -> \
            tuple[tuple[float, float], tuple[float, float]]:
        transformer = pyproj.Transformer.from_crs(
            grid_mapping.crs,
            CRS_CRS84,
            always_xy=True
        )
        bbox = grid_mapping.xy_bbox
        # (x1, x2), (y1, y2)
        return transformer.transform((bbox[0], bbox[2]), (bbox[1], bbox[3]))

    (x1, x2), (y1, y2) = transform_bbox(grid_mapping)

    # TODO: Prefer original storage location.
    #       The "s3" operation is default.
    default_storage_url = f"{base_url}/s3/datasets"

    return {
        "stac_version": STAC_VERSION,
        "stac_extensions": STAC_EXTENSIONS,
        "type": "Feature",
        "id": dataset_id,
        "bbox": [x1, y1, x2, y2],
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]],
            ],
        },
        "properties": cube_properties,
        "collection": collection_id,
        "links": [
            _root_link(base_url),
            {
                "rel": "self",
                "href": f"{base_url}{PATH_PREFIX}/collections/{collection_id}"
                        f"/items/{dataset_id}"
            },
            {
                "rel": "collection",
                "href": f"{base_url}{PATH_PREFIX}/collections/{collection_id}"
            },
            {
                "rel": "parent",
                "href": f"{base_url}{PATH_PREFIX}/collections/{collection_id}"
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
                    for v in xcube_data_vars
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
                    for v in xcube_data_vars
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


def _get_time_properties(dataset):
    if 'time' in dataset:
        time_var = dataset['time']
        start_time = to_json_value(time_var[0])
        end_time = to_json_value(time_var[-1])
        time_properties = {
            'datetime': start_time
        } if start_time == end_time else {
            'datetime': None,
            'start_datetime': start_time,
            'end_datetime': end_time
        }
    else:
        time_properties = {
            # TODO Decide what to use as a fall-back datetime
            'datetime': '2000-01-01T00:00:00Z'
        }
    return time_properties


def _get_xc_variables(variables: Mapping[Hashable, xr.DataArray]) \
        -> List[Dict[str, Any]]:
    """Create the value of the "xcube:coords" or
    "xcube:data_vars" property for the given *dataset*.
    """
    return [_get_xc_variable(var_name, var)
            for var_name, var in variables.items()]


def _get_xc_variable(var_name: Hashable, var: xr.DataArray) -> Dict[str, Any]:
    """Create an entry of the value of the "xcube:coords" or
    "xcube:data_vars" property for the given *dataset*.
    """
    return {
        "name": str(var_name),
        "dtype": str(var.dtype),
        "dims": to_json_value(var.dims),
        "chunks": to_json_value(var.chunks) if var.chunks else None,
        "shape": to_json_value(var.shape),
        "attrs": to_json_value(var.attrs),
        # "encoding": to_json_value(var.encoding),
    }


def _get_dc_dimensions(dataset: xr.Dataset,
                       grid_mapping: GridMapping) -> Dict[str, Any]:
    """Create the value of the "datacube:dimensions" property
    for the given *dataset*.
    """
    x_dim_name, y_dim_name = grid_mapping.xy_dim_names
    x_var_name, y_var_name = grid_mapping.xy_var_names
    dc_dimensions = {
        x_dim_name: _get_dc_spatial_dimension(
            dataset[x_var_name], "x", grid_mapping,
        ),
        y_dim_name: _get_dc_spatial_dimension(
            dataset[y_var_name], "y", grid_mapping,
        ),
    }
    if "time" in dataset.dims \
            and "time" in dataset.coords \
            and dataset["time"].ndim == 1:
        dc_dimensions.update(
            time=_get_dc_temporal_dimension(dataset["time"])
        )
    for dim_name in dataset.dims.keys():
        if dim_name not in {x_dim_name, y_dim_name, "time"} \
                and dim_name in dataset:
            dc_dimensions.update(
                dim_name=_get_dc_additional_dimension(dataset[dim_name])
            )
    return dc_dimensions


def _get_dc_spatial_dimension(
        var: xr.DataArray,
        axis: str,
        grid_mapping: GridMapping
) -> Dict[str, Any]:
    """Create a spatial dimension of the "datacube:dimensions" property
    for the given *var* and *axis*.
    """
    asset = _get_dc_dimension(var, "spatial", axis=axis)
    if axis == "x":
        extent = grid_mapping.x_min, grid_mapping.x_max
        step = grid_mapping.x_res if grid_mapping.is_regular else None
    else:
        extent = grid_mapping.y_min, grid_mapping.y_max
        step = grid_mapping.y_res if grid_mapping.is_regular else None
    asset["extent"] = to_json_value(extent)
    asset["step"] = to_json_value(step)
    asset["reference_system"] = to_json_value(grid_mapping.crs.srs)
    return asset


def _get_dc_temporal_dimension(
        var: xr.DataArray
) -> Dict[str, Any]:
    """Create a temporal dimension of the "datacube:dimensions" property
    for the given time *var*.
    """
    asset = _get_dc_dimension(var, "temporal", axis=None,
                              drop_unit=True)
    asset["values"] = [to_json_value(t) for t in var.values]
    return asset


def _get_dc_additional_dimension(
        var: xr.DataArray,
        type: str = "unknown"
) -> Dict[str, Any]:
    """Create an additional dimension of the "datacube:dimensions" property
    for the given *var* and *type*.
    """
    asset = _get_dc_dimension(var, type, axis=None)
    if var.ndim == 1:
        asset["range"] = [to_json_value(var[0]), to_json_value(var[-1])]
        if var.size > 1:
            diff_var = np.diff(var)
            if np.issubdtype(var.dtype, np.number) \
                    and np.allclose(np.diff(diff_var), 0):
                asset["step"] = to_json_value(diff_var[0])
        if "step" not in asset and var.size < _MAX_NUM_VALUES:
            asset["values"] = [to_json_value(t) for t in var.values]
    return asset


def _get_dc_dimension(
        var: xr.DataArray,
        type: str,
        axis: Optional[str] = None,
        drop_unit: bool = False
) -> Dict[str, Any]:
    """Create a generic dimension of the "datacube:dimensions" property
    for the given *var*, *type*, and optional *axis*.
    """
    asset = dict(type=type)
    if axis is not None:
        asset.update(axis=axis)
    _set_dc_description(asset, var)
    if not drop_unit:
        _set_dc_unit(asset, var)
    return asset


def _get_dc_variables(dataset: xr.Dataset, dc_dimensions):
    """Create the value of the "datacube:variables" property
    for the given *dataset*.
    """
    return dict(
        **__get_dc_variables(dataset.data_vars, "data", dc_dimensions),
        **__get_dc_variables(dataset.coords, "auxiliary", dc_dimensions),
    )


def __get_dc_variables(variables: Mapping[Hashable, xr.DataArray],
                       type: str,
                       dc_dimensions: Dict[str, Any]):
    """Create a partial value of the "datacube:variables" property
    for the given *variables* and *type*.
    """
    return {
        str(var_name): _get_dc_variable(var, type)
        for var_name, var in variables.items()
        if var_name not in dc_dimensions and var.ndim >= 1
    }


def _get_dc_variable(
        var: xr.DataArray,
        type: str
) -> Dict[str, Any]:
    """Create a generic variable of the "datacube:variables" property
    for the given *var*, *type*, and optional *axis*.
    """
    asset = dict(type=type, dimensions=list(var.dims))
    _set_dc_description(asset, var)
    _set_dc_unit(asset, var)
    return asset


def _set_dc_description(asset, var):
    """Set the "description" property of given asset, if any."""
    description = _get_str_attr(var.attrs,
                                ['description', 'title', 'long_name'])
    if description:
        asset.update(description=description)


def _set_dc_unit(asset, var):
    """Set the "unit" property of given asset, if any."""
    unit = _get_str_attr(var.attrs, ['unit', 'units'])
    if unit:
        asset.update(unit=unit)


def _get_str_attr(attrs: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        v = attrs.get(k)
        if isinstance(v, str) and v:
            return v
    return None


def _assert_valid_collection(ctx: DatasetsContext, collection_id: str):
    # c_id, _, _ = _get_collection_metadata(ctx.config)
    collection_ids = [c['Identifier'] for c in ctx.get_dataset_configs()]
    if (
        collection_id not in collection_ids
        and collection_id != DEFAULT_COLLECTION_ID
    ):
        raise ApiError.NotFound(f'Collection "{collection_id}" not found')


def _get_catalog_metadata(config: ServerConfig):
    stac_config = config.get("STAC", {})
    catalog_id = stac_config.get(
        "Identifier", DEFAULT_CATALOG_ID
    )
    catalog_title = stac_config.get(
        "Title", DEFAULT_CATALOG_TITLE
    )
    catalog_description = stac_config.get(
        "Description", DEFAULT_CATALOG_DESCRIPTION
    )
    return catalog_id, catalog_title, catalog_description


def _get_collection_metadata(config: ServerConfig):
    stac_config = config.get("STAC", {})
    collection_config = stac_config.get("Collection", {})
    collection_id = collection_config.get(
        "Identifier", DEFAULT_COLLECTION_ID
    )
    collection_title = collection_config.get(
        "Title", DEFAULT_COLLECTION_TITLE
    )
    collection_description = collection_config.get(
        "Description", DEFAULT_COLLECTION_DESCRIPTION
    )
    return collection_id, collection_title, collection_description


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
