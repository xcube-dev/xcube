# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonStringSchema

BOOLEAN_SCHEMA = JsonBooleanSchema()
NUMBER_SCHEMA = JsonNumberSchema()
URI_SCHEMA = JsonStringSchema(format="uri")
IDENTIFIER_SCHEMA = JsonStringSchema(min_length=1)
CHUNK_SIZE_SCHEMA = JsonStringSchema(min_length=2)  # TODO: use pattern
STRING_SCHEMA = JsonStringSchema()
PATH_SCHEMA = JsonStringSchema(min_length=1)

BOUNDING_BOX_ITEMS = [NUMBER_SCHEMA, NUMBER_SCHEMA, NUMBER_SCHEMA, NUMBER_SCHEMA]

BOUNDING_BOX_SCHEMA = JsonArraySchema(
    items=BOUNDING_BOX_ITEMS,
    description="Spatial bounding box given as [x-min, y-min, x-max, y-max].",
)

GEO_BOUNDING_BOX_SCHEMA = JsonArraySchema(
    items=BOUNDING_BOX_ITEMS,
    description="Spatial bounding box given as [x-min, y-min, x-max, y-max]"
    " using geographical coordinates.",
)

FILE_SYSTEM_SCHEMA = JsonStringSchema(enum=["memory", "obs", "local", "s3", "file"])
