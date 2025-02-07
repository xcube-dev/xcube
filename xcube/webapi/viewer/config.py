# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.util.jsonschema import JsonArraySchema, JsonObjectSchema
from xcube.webapi.common.schemas import STRING_SCHEMA

CONFIGURATION_SCHEMA = JsonObjectSchema(
    properties=dict(
        Path=STRING_SCHEMA,
    ),
    additional_properties=False,
)

PERSISTENCE_SCHEMA = JsonObjectSchema(
    properties=dict(
        Path=STRING_SCHEMA, StorageOptions=JsonObjectSchema(additional_properties=True)
    ),
    required=["Path"],
    additional_properties=False,
)

EXTENSIONS_SCHEMA = JsonArraySchema(
    items=STRING_SCHEMA,
    min_items=1,
)

AUGMENTATION_SCHEMA = JsonObjectSchema(
    properties=dict(
        Path=STRING_SCHEMA,
        Extensions=EXTENSIONS_SCHEMA,
    ),
    required=["Extensions"],
    additional_properties=False,
)

VIEWER_SCHEMA = JsonObjectSchema(
    properties=dict(
        Configuration=CONFIGURATION_SCHEMA,
        Augmentation=AUGMENTATION_SCHEMA,
        Persistence=PERSISTENCE_SCHEMA,
    ),
    additional_properties=False,
)

CONFIG_SCHEMA = JsonObjectSchema(
    properties=dict(
        Viewer=VIEWER_SCHEMA,
    )
)
