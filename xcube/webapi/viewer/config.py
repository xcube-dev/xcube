# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.util.jsonschema import JsonObjectSchema
from xcube.webapi.common.schemas import STRING_SCHEMA

CONFIGURATION_SCHEMA = JsonObjectSchema(
    properties=dict(
        Path=STRING_SCHEMA,
    ),
    additional_properties=False,
)

VIEWER_SCHEMA = JsonObjectSchema(
    properties=dict(
        Configuration=CONFIGURATION_SCHEMA,
    ),
    additional_properties=False,
)

CONFIG_SCHEMA = JsonObjectSchema(
    properties=dict(
        Viewer=VIEWER_SCHEMA,
    )
)
