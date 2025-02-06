# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.util.jsonschema import JsonIntegerSchema, JsonObjectSchema

COMPUTE_CONFIG_SCHEMA = JsonObjectSchema(
    properties=dict(
        MaxWorkers=JsonIntegerSchema(minimum=1),
        # Executor=JsonObjectSchema(),
        # OpRegistry=JsonStringSchema(),
    ),
    additional_properties=False,
)

CONFIG_SCHEMA = JsonObjectSchema(
    properties=dict(
        Compute=COMPUTE_CONFIG_SCHEMA,
    )
)
