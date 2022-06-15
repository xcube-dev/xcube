# The MIT License (MIT)
# Copyright (c) 2022 by the xcube team and contributors
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

from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.webapi.config import BooleanSchema
from xcube.webapi.config import BoundingBoxSchema
from xcube.webapi.config import ChunkSizeSchema
from xcube.webapi.config import FileSystemSchema
from xcube.webapi.config import IdentifierSchema
from xcube.webapi.config import PathSchema
from xcube.webapi.config import StringSchema
from xcube.webapi.config import UrlSchema
from ..places.config import PLACE_GROUP_SCHEMA

AUGMENTATION_SCHEMA = JsonObjectSchema(
    properties=dict(
        Path=PathSchema,
        Function=IdentifierSchema,
        InputParameters=JsonObjectSchema(
            additional_properties=True,
        ),
    ),
    required=[
        'Path',
        'Function',
    ],
    additional_properties=False,
)

ACCESS_CONTROL_SCHEMA = JsonObjectSchema(
    properties=dict(
        IsSubstitute=BooleanSchema,
        RequiredScopes=JsonArraySchema(items=IdentifierSchema)
    ),
    additional_properties=False,
)

COMMON_DATASET_PROPERTIES = dict(
    Title=StringSchema,
    TimeSeriesDataset=IdentifierSchema,
    BoundingBox=BoundingBoxSchema,
    ChunkCacheSize=ChunkSizeSchema,
    Augmentation=AUGMENTATION_SCHEMA,
    Style=IdentifierSchema,
    Hidden=BooleanSchema,
    AccessControl=ACCESS_CONTROL_SCHEMA,
    PlaceGroups=JsonArraySchema(items=PLACE_GROUP_SCHEMA),
)

DATASET_CONFIG_SCHEMA = JsonObjectSchema(
    properties=dict(
        Identifier=IdentifierSchema,
        Path=PathSchema,
        FileSystem=FileSystemSchema,
        Anonymous=BooleanSchema,
        Endpoint=UrlSchema,
        Region=IdentifierSchema,
        Function=IdentifierSchema,
        InputDatasets=JsonArraySchema(items=IdentifierSchema),
        InputParameters=JsonObjectSchema(additional_properties=True),
        **COMMON_DATASET_PROPERTIES,
    ),
    required=[
        'Identifier',
        'Path'
    ],
    additional_properties=False,
)

DATA_STORE_DATASET_SCHEMA = JsonObjectSchema(
    required=[
        'Path'
    ],
    properties=dict(
        Identifier=IdentifierSchema,
        Path=PathSchema,
        StoreInstanceId=IdentifierSchema,  # will be set by server
        StoreOpenParams=JsonObjectSchema(additional_properties=True),
        **COMMON_DATASET_PROPERTIES
    ),
    additional_properties=False,
)

DATA_STORE_SCHEMA = JsonObjectSchema(
    properties=dict(
        Identifier=IdentifierSchema,
        StoreId=IdentifierSchema,
        StoreParams=JsonObjectSchema(additional_properties=True),
        Datasets=JsonArraySchema(items=DATA_STORE_DATASET_SCHEMA),
    ),
    required=[
        'Identifier',
        'StoreId',
    ],
    additional_properties=False,
)

ValueRangeSchema = JsonArraySchema(items=[
    JsonNumberSchema(),
    JsonNumberSchema()
])

COLOR_MAPPING_SCHEMA = JsonObjectSchema(
    properties=dict(
        ColorBar=StringSchema,
        ValueRange=ValueRangeSchema
    ),
    required=[
        "ValueRange",
        "ColorBar"
    ],
    additional_properties=False
)

CHANNEL_MAPPING_SCHEMA = JsonObjectSchema(
    properties=dict(
        ValueRange=JsonArraySchema(items=[
            JsonNumberSchema(),
            JsonNumberSchema()
        ]),
        Variable=StringSchema
    ),
    required=[
        "ValueRange",
        "Variable"
    ],
    additional_properties=False
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
    additional_properties=False
)

STYLE_SCHEMA = JsonObjectSchema(
    properties=dict(
        Identifier=StringSchema,
        ColorMappings=JsonObjectSchema(
            properties=dict(
                rgb=RGB_MAPPING_SCHEMA
            ),
            additional_properties=COLOR_MAPPING_SCHEMA
        )
    ),
    required=[
        'Identifier',
        'ColorMappings',
    ],
    additional_properties=False
)

SERVICE_PROVIDER_SCHEMA = JsonObjectSchema(
    additional_properties=True,
)

CONFIG_SCHEMA = JsonObjectSchema(
    properties=dict(
        DatasetAttribution=JsonArraySchema(items=StringSchema),
        AccessControl=ACCESS_CONTROL_SCHEMA,
        DatasetChunkCacheSize=ChunkSizeSchema,
        Datasets=JsonArraySchema(items=DATASET_CONFIG_SCHEMA),
        DataStores=JsonArraySchema(items=DATA_STORE_SCHEMA),
        Styles=JsonArraySchema(items=STYLE_SCHEMA),
        ServiceProvider=SERVICE_PROVIDER_SCHEMA,
    )
)
