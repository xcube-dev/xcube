# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.util.jsonschema import JsonIntegerSchema
from xcube.util.jsonschema import JsonObjectSchema


DEFAULT_MAX_VOXEL_COUNT = 256**3

VOLUMES_ACCESS_SCHEMA = JsonObjectSchema(
    properties=dict(
        MaxVoxelCount=JsonIntegerSchema(
            minimum=10**3,
            default=DEFAULT_MAX_VOXEL_COUNT,
        ),
    ),
    additional_properties=False,
)

CONFIG_SCHEMA = JsonObjectSchema(
    properties=dict(
        VolumesAccess=VOLUMES_ACCESS_SCHEMA,
    )
)
