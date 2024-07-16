# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonComplexSchema
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.webapi.common.schemas import BOOLEAN_SCHEMA
from xcube.webapi.common.schemas import CHUNK_SIZE_SCHEMA
from xcube.webapi.common.schemas import FILE_SYSTEM_SCHEMA
from xcube.webapi.common.schemas import GEO_BOUNDING_BOX_SCHEMA
from xcube.webapi.common.schemas import IDENTIFIER_SCHEMA
from xcube.webapi.common.schemas import PATH_SCHEMA
from xcube.webapi.common.schemas import STRING_SCHEMA
from xcube.webapi.common.schemas import URI_SCHEMA
from ..places.config import PLACE_GROUP_SCHEMA

ATTRIBUTION_SCHEMA = JsonComplexSchema(
    one_of=[
        STRING_SCHEMA,
        JsonArraySchema(items=STRING_SCHEMA),
    ]
)

VARIABLES_SCHEMA = JsonArraySchema(
    items=IDENTIFIER_SCHEMA,
    min_items=1,
    description="Names of variables to be published."
    " Names may use wildcard characters '*' and '?'."
    " Also determines the order of variables.",
)

VALUE_RANGE_SCHEMA = JsonArraySchema(items=[JsonNumberSchema(), JsonNumberSchema()])

AUGMENTATION_SCHEMA = JsonObjectSchema(
    properties=dict(
        Path=PATH_SCHEMA,
        Function=IDENTIFIER_SCHEMA,
        Class=IDENTIFIER_SCHEMA,
        InputParameters=JsonObjectSchema(
            additional_properties=True,
        ),
    ),
    required=[
        "Path",
        "Function",
    ],
    additional_properties=False,
)

ACCESS_CONTROL_SCHEMA = JsonObjectSchema(
    properties=dict(
        IsSubstitute=BOOLEAN_SCHEMA,
        RequiredScopes=JsonArraySchema(items=IDENTIFIER_SCHEMA),
    ),
    additional_properties=False,
)

COMMON_DATASET_PROPERTIES = dict(
    Title=STRING_SCHEMA,
    GroupTitle=STRING_SCHEMA,
    Tags=JsonArraySchema(items=STRING_SCHEMA),
    Variables=VARIABLES_SCHEMA,
    TimeSeriesDataset=IDENTIFIER_SCHEMA,
    BoundingBox=GEO_BOUNDING_BOX_SCHEMA,
    ChunkCacheSize=CHUNK_SIZE_SCHEMA,
    Augmentation=AUGMENTATION_SCHEMA,
    Style=IDENTIFIER_SCHEMA,
    Hidden=BOOLEAN_SCHEMA,
    AccessControl=ACCESS_CONTROL_SCHEMA,
    PlaceGroups=JsonArraySchema(items=PLACE_GROUP_SCHEMA),
    Attribution=ATTRIBUTION_SCHEMA,
)

DATASET_CONFIG_SCHEMA = JsonObjectSchema(
    properties=dict(
        Identifier=IDENTIFIER_SCHEMA,
        StoreInstanceId=IDENTIFIER_SCHEMA,  # will be set by server
        Path=PATH_SCHEMA,
        FileSystem=FILE_SYSTEM_SCHEMA,
        Anonymous=BOOLEAN_SCHEMA,
        Endpoint=URI_SCHEMA,
        Region=IDENTIFIER_SCHEMA,
        Function=IDENTIFIER_SCHEMA,
        Class=IDENTIFIER_SCHEMA,
        InputDatasets=JsonArraySchema(items=IDENTIFIER_SCHEMA),
        InputParameters=JsonObjectSchema(additional_properties=True),
        **COMMON_DATASET_PROPERTIES,
    ),
    required=["Identifier", "Path"],
    additional_properties=False,
)

DATA_STORE_DATASET_SCHEMA = JsonObjectSchema(
    required=["Path"],
    properties=dict(
        Identifier=IDENTIFIER_SCHEMA,
        Path=PATH_SCHEMA,
        StoreInstanceId=IDENTIFIER_SCHEMA,  # will be set by server
        StoreOpenParams=JsonObjectSchema(additional_properties=True),
        **COMMON_DATASET_PROPERTIES,
    ),
    additional_properties=False,
)

DATA_STORE_SCHEMA = JsonObjectSchema(
    properties=dict(
        Identifier=IDENTIFIER_SCHEMA,
        StoreId=IDENTIFIER_SCHEMA,
        StoreParams=JsonObjectSchema(additional_properties=True),
        Datasets=JsonArraySchema(items=DATA_STORE_DATASET_SCHEMA),
    ),
    required=[
        "Identifier",
        "StoreId",
    ],
    additional_properties=False,
)

COLOR_MAPPING_EXPLICIT_SCHEMA = JsonObjectSchema(
    properties=dict(ColorBar=STRING_SCHEMA, ValueRange=VALUE_RANGE_SCHEMA),
    required=[],
    additional_properties=False,
)

COLOR_MAPPING_BY_PATH_SCHEMA = JsonObjectSchema(
    properties=dict(
        ColorFile=STRING_SCHEMA,
    ),
    required=[
        "ColorFile",
    ],
    additional_properties=False,
)

COLOR_MAPPING_SCHEMA = JsonComplexSchema(
    one_of=[
        COLOR_MAPPING_EXPLICIT_SCHEMA,
        COLOR_MAPPING_BY_PATH_SCHEMA,
    ]
)

CHANNEL_MAPPING_SCHEMA = JsonObjectSchema(
    properties=dict(
        ValueRange=JsonArraySchema(items=[JsonNumberSchema(), JsonNumberSchema()]),
        Variable=STRING_SCHEMA,
    ),
    required=["ValueRange", "Variable"],
    additional_properties=False,
)

RGB_MAPPING_SCHEMA = JsonObjectSchema(
    properties=dict(
        Red=CHANNEL_MAPPING_SCHEMA,
        Green=CHANNEL_MAPPING_SCHEMA,
        Blue=CHANNEL_MAPPING_SCHEMA,
    ),
    required=[
        "Red",
        "Green",
        "Blue",
    ],
    additional_properties=False,
)

STYLE_SCHEMA = JsonObjectSchema(
    properties=dict(
        Identifier=STRING_SCHEMA,
        ColorMappings=JsonObjectSchema(
            properties=dict(rgb=RGB_MAPPING_SCHEMA),
            additional_properties=COLOR_MAPPING_SCHEMA,
        ),
    ),
    required=[
        "Identifier",
        "ColorMappings",
    ],
    additional_properties=False,
)

SERVICE_PROVIDER_SCHEMA = JsonObjectSchema(
    additional_properties=True,
)

CONFIG_SCHEMA = JsonObjectSchema(
    properties=dict(
        DatasetAttribution=ATTRIBUTION_SCHEMA,
        AccessControl=ACCESS_CONTROL_SCHEMA,
        DatasetChunkCacheSize=CHUNK_SIZE_SCHEMA,
        Datasets=JsonArraySchema(items=DATASET_CONFIG_SCHEMA),
        DataStores=JsonArraySchema(items=DATA_STORE_SCHEMA),
        Styles=JsonArraySchema(items=STYLE_SCHEMA),
        ServiceProvider=SERVICE_PROVIDER_SCHEMA,
    )
)
