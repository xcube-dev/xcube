# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.util.jsonschema import JsonObjectSchema, JsonStringSchema

DEFAULT_CATALOG_ID = "xcube-server"
DEFAULT_CATALOG_TITLE = "xcube Server"
DEFAULT_CATALOG_DESCRIPTION = "Catalog of datasets served by xcube."

# ID, name, and description for the unified collection containing a feature
# for each datacube
DEFAULT_COLLECTION_ID = "datacubes"
DEFAULT_COLLECTION_TITLE = "Data cubes"
DEFAULT_COLLECTION_DESCRIPTION = "a collection of xcube datasets"

# As well as the unified collection, there's an individual collection for
# each datacube, with the same name as that datacube, and containing a single
# feature representing that datacube. This is the name of that single feature.
DEFAULT_FEATURE_ID = "datacube"

# Prefix for STAC and OGC endpoints
PATH_PREFIX = "/ogc"

COLLECTION_SCHEMA = JsonObjectSchema(
    properties=dict(
        Identifier=JsonStringSchema(default=DEFAULT_COLLECTION_ID),
        Title=JsonStringSchema(default=DEFAULT_COLLECTION_TITLE),
        Description=JsonStringSchema(default=DEFAULT_COLLECTION_DESCRIPTION),
    ),
    additional_properties=False,
    required=[],
)

STAC_SCHEMA = JsonObjectSchema(
    properties=dict(
        Identifier=JsonStringSchema(default=DEFAULT_CATALOG_ID),
        Title=JsonStringSchema(default=DEFAULT_CATALOG_TITLE),
        Description=JsonStringSchema(default=DEFAULT_CATALOG_DESCRIPTION),
        Collection=COLLECTION_SCHEMA,
    ),
    additional_properties=False,
    required=[],
)

CONFIG_SCHEMA = JsonObjectSchema(
    properties=dict(
        STAC=STAC_SCHEMA,
    )
)
