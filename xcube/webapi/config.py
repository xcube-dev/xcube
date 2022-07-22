# TODO (forman): xcube Server NG: remove this module, must no longer be used

from abc import ABC

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
FileSystemSchema = JsonStringSchema(
    enum=['memory', 'obs', 'local', 's3', 'file']
)


class _ConfigObject(JsonObject, ABC):
    def __init__(self, **kwargs):
        self._inject_attrs(kwargs)


class ServiceConfig(_ConfigObject):

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            factory=ServiceConfig,
            properties=dict(
                Authentication=Authentication.get_schema(),
                DatasetAttribution=JsonArraySchema(items=StringSchema),
                DatasetChunkCacheSize=ChunkSizeSchema,
                Datasets=JsonArraySchema(items=DatasetConfig.get_schema()),
                DataStores=JsonArraySchema(items=DataStoreConfig.get_schema()),
                PlaceGroups=JsonArraySchema(items=PlaceGroupConfig.get_schema()),
                Styles=JsonArraySchema(items=StyleConfig.get_schema()),
                ServiceProvider=ServiceProvider.get_schema(),
            ),
            additional_properties=False,
        )


class Authentication(_ConfigObject):
    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            factory=Authentication,
            required=[
                'Authority',
                'Audience',
            ],
            properties=dict(
                Authority=JsonStringSchema(),
                Domain=JsonStringSchema(),
                Audience=UrlSchema,
                Algorithms=JsonArraySchema(items=IdentifierSchema),
            ),
            additional_properties=False,
        )


def _get_common_dataset_properties():
    return dict(
        Title=StringSchema,
        TimeSeriesDataset=IdentifierSchema,
        BoundingBox=BoundingBoxSchema,
        ChunkCacheSize=ChunkSizeSchema,
        Augmentation=Augmentation.get_schema(),
        Style=IdentifierSchema,
        Hidden=BooleanSchema,
        AccessControl=AccessControl.get_schema(),
        PlaceGroups=JsonArraySchema(items=JsonObjectSchema(
            properties=dict(
                PlaceGroupRef=IdentifierSchema,
            ),
        )),
    )


class DatasetConfig(_ConfigObject):
    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            factory=DatasetConfig,
            required=[
                'Identifier',
                'Path'
            ],
            properties=dict(
                Identifier=IdentifierSchema,
                Path=PathSchema,
                FileSystem=FileSystemSchema,
                Anonymous=BooleanSchema,
                Endpoint=UrlSchema,
                Region=IdentifierSchema,
                Function=IdentifierSchema,
                InputDatasets=JsonArraySchema(items=IdentifierSchema),
                InputParameters=JsonObjectSchema(
                    additional_properties=True,
                ),
                **_get_common_dataset_properties(),
            ),
            additional_properties=False,
        )


class Augmentation(_ConfigObject):
    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            factory=Authentication,
            required=[
                'Path',
                'Function',
            ],
            properties=dict(
                Path=PathSchema,
                Function=IdentifierSchema,
                InputParameters=JsonObjectSchema(
                    additional_properties=True,
                ),
            ),
            additional_properties=False,
        )


class PlaceGroupConfig(_ConfigObject):
    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            factory=PlaceGroupConfig,
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
            ),
            additional_properties=False,
        )


class DataStoreConfig(_ConfigObject):
    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            factory=DataStoreConfig,
            required=[
                'Identifier',
                'StoreId',
            ],
            properties=dict(
                Identifier=IdentifierSchema,
                StoreId=IdentifierSchema,
                StoreParams=JsonObjectSchema(
                    additional_properties=True,
                ),
                Datasets=JsonArraySchema(
                    items=DataStoreDatasetConfig.get_schema(),
                ),
            ),
            additional_properties=False,
        )


class DataStoreDatasetConfig(_ConfigObject):
    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            factory=DataStoreDatasetConfig,
            required=[
                'Path'
            ],
            properties=dict(
                Identifier=IdentifierSchema,
                Path=PathSchema,
                StoreInstanceId=IdentifierSchema,  # will be set by server
                StoreOpenParams=JsonObjectSchema(additional_properties=True),
                **_get_common_dataset_properties()
            ),
            additional_properties=False,
        )


class PlaceGroupJoin(_ConfigObject):
    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            factory=PlaceGroupJoin,
            required=[
                'Property',
                'Path',
            ],
            properties=dict(
                Property=IdentifierSchema,
                Path=PathSchema,
            ),
            additional_properties=False,
        )


class StyleConfig(_ConfigObject):
    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            factory=StyleConfig,
            required=[
                'Identifier',
                'ColorMappings',
            ],
            properties=dict(
                Identifier=JsonStringSchema(min_length=1),
                ColorMappings=JsonObjectSchema(
                    additional_properties=ColorMapping.get_schema()
                )
            ),
            additional_properties=False,
        )


class ColorMapping(_ConfigObject):
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
            ),
            additional_properties=False,
        )


class ServiceProvider(_ConfigObject):
    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            additional_properties=True,
        )


class AccessControl(_ConfigObject):
    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                IsSubstitute=JsonBooleanSchema(),
                RequiredScopes=JsonArraySchema(items=IdentifierSchema)
            ),
            additional_properties=False,
        )
