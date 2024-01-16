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

from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema

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
DEFAULT_FEATURE_ID = 'datacube'

# Prefix for STAC and OGC endpoints
PATH_PREFIX = '/ogc'

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
