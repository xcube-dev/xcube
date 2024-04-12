# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.webapi.common.schemas import IDENTIFIER_SCHEMA
from xcube.webapi.common.schemas import PATH_SCHEMA
from xcube.webapi.common.schemas import STRING_SCHEMA

PLACE_GROUP_JOIN_SCHEMA = JsonObjectSchema(
    properties=dict(
        Property=IDENTIFIER_SCHEMA,
        Path=PATH_SCHEMA,
    ),
    required=[
        "Property",
        "Path",
    ],
    additional_properties=False,
)

PLACE_GROUP_SCHEMA = JsonObjectSchema(
    properties=dict(
        Identifier=IDENTIFIER_SCHEMA,
        Title=STRING_SCHEMA,
        Path=PATH_SCHEMA,
        Query=STRING_SCHEMA,
        Join=PLACE_GROUP_JOIN_SCHEMA,
        CharacterEncoding=STRING_SCHEMA,
        PropertyMapping=JsonObjectSchema(additional_properties=PATH_SCHEMA),
        PlaceGroupRef=IDENTIFIER_SCHEMA,
    ),
    required=[
        # Either we have
        #   'Identifier',
        #   'Path',
        # or we have
        #   'Identifier',
        #   'Query',
        # or we must specify
        #   'PlaceGroupRef',
    ],
    additional_properties=False,
)

CONFIG_SCHEMA = JsonObjectSchema(
    properties=dict(PlaceGroups=JsonArraySchema(items=PLACE_GROUP_SCHEMA))
)
