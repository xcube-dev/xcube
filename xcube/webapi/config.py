from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonObject
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema

BooleanSchema = JsonBooleanSchema()
NumberSchema = JsonNumberSchema()
UrlSchema = JsonStringSchema(format='uri')
IdentifierSchema = JsonStringSchema(min_length=1)
ChunkSizeSchema = JsonStringSchema(min_length=2)  # TODO: use pattern
StringSchema = JsonStringSchema()
PathSchema = JsonStringSchema(min_length=1)
BoundingBoxSchema = JsonArraySchema(items=[
    NumberSchema,
    NumberSchema,
    NumberSchema,
    NumberSchema
])
FileSystemSchema = JsonStringSchema(enum=['memory', 'obs', 'local'])


class ServiceConfig(JsonObject):
    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                Authentication=Authentication.get_schema(),
                DatasetAttribution=JsonArraySchema(items=StringSchema),
                DatasetChunkCacheSize=ChunkSizeSchema,
                Datasets=JsonArraySchema(items=DatasetConfig.get_schema()),
                DataStores=JsonArraySchema(items=DataStoreConfig.get_schema()),
                PlaceGroups=JsonArraySchema(items=PlaceGroupConfig.get_schema()),
                Styles=JsonArraySchema(items=StyleConfig.get_schema()),
                ServiceProvider=ServiceProvider.get_schema(),
            )
        )


class Authentication(JsonObject):
    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            required=[
                'Domain',
                'Audience',
            ],
            properties=dict(
                Domain=JsonStringSchema(),
                Audience=UrlSchema,
            )
        )


class DatasetConfig(JsonObject):
    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            required=[
                'Identifier',
                'Path',
            ],
            properties=dict(
                Identifier=IdentifierSchema,
                TimeSeriesDataset=IdentifierSchema,
                Title=StringSchema,
                Path=PathSchema,
                FileSystem=FileSystemSchema,
                Anonymous=BooleanSchema,
                Endpoint=UrlSchema,
                Region=IdentifierSchema,
                BoundingBox=BoundingBoxSchema,
                Function=IdentifierSchema,
                InputDatasets=JsonArraySchema(items=IdentifierSchema),
                InputParameters=JsonObjectSchema(
                    additional_properties=True,
                ),
                ChunkCacheSize=ChunkSizeSchema,
                Style=IdentifierSchema,
                Hidden=BooleanSchema,
                AccessControl=AccessControl.get_schema(),
                PlaceGroups=JsonArraySchema(items=JsonObjectSchema(
                    properties=dict(
                        PlaceGroupRef=IdentifierSchema,
                    ),
                )),
            )
        )


class PlaceGroupConfig(JsonObject):
    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            required=[
                'Identifier',
                'Path',
            ],
            properties=dict(
                Identifier=IdentifierSchema,
                Title=StringSchema,
                Path=PathSchema,
                Join=PlaceGroupJoin.get_schema(),
                PropertyMapping=JsonObjectSchema(
                    additional_properties=PathSchema,
                ),
            )
        )


class DataStoreConfig(JsonObject):
    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            required=[
                'Identifier',
                'Path',
            ],
            properties=dict(
                Identifier=IdentifierSchema,
                Params=JsonObjectSchema(
                    additional_properties=True,
                ),
                Datasets=JsonArraySchema(
                    items=DataStoreDataConfig.get_schema()
                ),
            )
        )


class DataStoreDataConfig(JsonObject):
    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            required=[
                'Identifier',
                'Path',
            ],
            properties=dict(
                TimeSeriesDataset=IdentifierSchema,
                ChunkCacheSize=ChunkSizeSchema,
                Style=IdentifierSchema,
                Hidden=BooleanSchema,
                AccessControl=AccessControl.get_schema(),
                PlaceGroups=JsonArraySchema(items=JsonObjectSchema(
                    properties=dict(
                        PlaceGroupRef=IdentifierSchema,
                    ),
                )),
            )
        )


class PlaceGroupJoin(JsonObject):
    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            required=[
                'Property',
                'Path',
            ],
            properties=dict(
                Property=IdentifierSchema,
                Path=PathSchema,
            )
        )


class StyleConfig(JsonObject):
    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            required=[
                'Identifier',
                'ColorMappings',
            ],
            properties=dict(
                Identifier=JsonStringSchema(min_length=1),
                ColorMappings=JsonObjectSchema(
                    additional_properties=ColorMapping.get_schema()
                )
            )
        )


class ColorMapping(JsonObject):
    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            required=[
                'ColorBar',
                'ValueRange',
            ],
            properties=dict(
                ColorBar=JsonStringSchema(min_length=1),
                ValueRange=JsonArraySchema(items=[
                    JsonNumberSchema(),
                    JsonNumberSchema()
                ])
            )
        )


class ServiceProvider(JsonObject):
    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            additional_properties=True,
        )


class AccessControl(JsonObject):
    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                IsSubstitute=JsonBooleanSchema(),
                RequiredScopes=JsonArraySchema(items=IdentifierSchema)
            )
        )
