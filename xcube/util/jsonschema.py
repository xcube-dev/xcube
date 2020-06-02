from abc import ABCMeta, abstractmethod
from typing import Dict, Any, Callable, Mapping, Sequence, Union

import jsonschema

from xcube.util.undefined import UNDEFINED

JsonToObj = Callable[..., Any]
ObjToJson = Callable[..., Any]

_TYPES_ENUM = {'null', 'boolean', 'integer', 'number', 'string', 'array', 'object'}
_NUMERIC_TYPES_ENUM = {'integer', 'number'}


class JsonSchema(metaclass=ABCMeta):

    # noinspection PyShadowingBuiltins
    def __init__(self,
                 type: str,
                 default: Any = UNDEFINED,
                 const: Any = UNDEFINED,
                 enum: Sequence[Any] = None,
                 nullable: bool = None,
                 title: str = None,
                 description: str = None,
                 json_to_obj: JsonToObj = None,
                 obj_to_json: ObjToJson = None):
        if type not in _TYPES_ENUM:
            raise ValueError(f'type must be one of {", ".join(_TYPES_ENUM)}')
        if json_to_obj is not None and not callable(json_to_obj):
            raise ValueError('json_to_obj must be callable')
        if obj_to_json is not None and not callable(obj_to_json):
            raise ValueError('obj_to_json must be callable')
        self.type = type
        self.default = default
        self.const = const
        self.enum = list(enum) if enum is not None else None
        self.nullable = nullable
        self.title = title
        self.description = description
        self.json_to_obj = json_to_obj
        self.obj_to_json = obj_to_json

    def to_dict(self) -> Dict[str, Any]:
        if self.nullable is not None and self.type != 'null':
            d = dict(type=[self.type, 'null'])
        else:
            d = dict(type=self.type)
        if self.default is not UNDEFINED:
            d.update(default=self.default)
        if self.const is not UNDEFINED:
            d.update(const=self.const)
        if self.enum is not None:
            d.update(enum=self.enum)
        if self.title is not None:
            d.update(title=self.title)
        if self.description is not None:
            d.update(description=self.description)
        return d

    def validate_instance(self, instance: Any):
        jsonschema.validate(instance=instance, schema=self.to_dict())

    @abstractmethod
    def to_json_instance(self, obj: Any) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def from_json_instance(self, instance: Any) -> Any:
        raise NotImplementedError()


class JsonSimpleTypeSchema(JsonSchema, metaclass=ABCMeta):
    # noinspection PyShadowingBuiltins
    def __init__(self, type: str, **kwargs):
        super().__init__(type, **kwargs)

    def to_json_instance(self, obj: Any) -> Any:
        return self.obj_to_json(obj) if self.obj_to_json is not None else obj

    def from_json_instance(self, instance: Any) -> Any:
        return self.json_to_obj(instance) if self.json_to_obj is not None else instance


class JsonNullSchema(JsonSimpleTypeSchema):
    def __init__(self, **kwargs):
        super().__init__(type='null', **kwargs)


class JsonBooleanSchema(JsonSimpleTypeSchema):
    def __init__(self, **kwargs):
        super().__init__(type='boolean', **kwargs)


class JsonStringSchema(JsonSimpleTypeSchema):
    # noinspection PyShadowingBuiltins
    def __init__(self,
                 format: str = None,
                 pattern: str = None,
                 min_length: int = None,
                 max_length: int = None,
                 **kwargs):
        super().__init__(type='string', **kwargs)
        self.format = format
        self.pattern = pattern
        self.min_length = min_length
        self.max_length = max_length

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.format is not None:
            d.update(format=self.format)
        if self.pattern is not None:
            d.update(pattern=self.pattern)
        if self.min_length is not None:
            d.update(minLength=self.min_length)
        if self.max_length is not None:
            d.update(maxLength=self.max_length)
        return d


class JsonNumberSchema(JsonSimpleTypeSchema):
    # noinspection PyShadowingBuiltins
    def __init__(self,
                 type: str = 'number',
                 minimum: Union[int, float] = None,
                 exclusive_minimum: Union[int, float] = None,
                 maximum: Union[int, float] = None,
                 exclusive_maximum: Union[int, float] = None,
                 multiple_of: Union[int, float] = None,
                 **kwargs):
        if type not in _NUMERIC_TYPES_ENUM:
            raise ValueError(f'Type must be one of {", ".join(_NUMERIC_TYPES_ENUM)}')
        super().__init__(type, **kwargs)
        self.minimum = minimum
        self.exclusive_minimum = exclusive_minimum
        self.maximum = maximum
        self.exclusive_maximum = exclusive_maximum
        self.multiple_of = multiple_of

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.minimum is not None:
            d.update(minimum=self.minimum)
        if self.exclusive_minimum is not None:
            d.update(exclusiveMinimum=self.exclusive_minimum)
        if self.maximum is not None:
            d.update(maximum=self.maximum)
        if self.exclusive_maximum is not None:
            d.update(exclusiveMaximum=self.exclusive_maximum)
        if self.multiple_of is not None:
            d.update(multipleOf=self.multiple_of)
        return d


class JsonIntegerSchema(JsonNumberSchema):
    def __init__(self, **kwargs):
        super().__init__(type='integer', **kwargs)


class JsonArraySchema(JsonSchema):

    def __init__(self,
                 items: Union[JsonSchema, Sequence[JsonSchema]] = None,
                 min_items: int = None,
                 max_items: int = None,
                 unique_items: bool = None,
                 **kwargs):
        super().__init__(type='array', **kwargs)
        self.items = items
        self.min_items = min_items
        self.max_items = max_items
        self.unique_items = unique_items

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.items is not None:
            if isinstance(self.items, JsonSchema):
                d.update(items=self.items.to_dict())
            else:
                d.update(items=[item.to_dict() for item in self.items])
        if self.min_items is not None:
            d.update(minItems=self.min_items)
        if self.max_items is not None:
            d.update(maxItems=self.max_items)
        if self.unique_items is not None:
            d.update(uniqueItems=self.unique_items)
        return d

    def to_json_instance(self, obj: Sequence[Any]) -> Sequence[Any]:
        if self.obj_to_json is not None:
            return self.obj_to_json(obj)
        return self._convert_instance(obj, 'to_json_instance')

    def from_json_instance(self, instance: Sequence[Any]) -> Any:
        new_instance = self._convert_instance(instance, 'from_json_instance')
        return self.json_to_obj(new_instance) if self.json_to_obj is not None else new_instance

    def _convert_instance(self, instance: Sequence[Any], method_name: str) -> Sequence[Any]:
        items_schema = self.items
        if isinstance(items_schema, JsonSchema):
            # Array is a validated list
            return [getattr(items_schema, method_name)(item) for item in instance]
        elif items_schema is not None:
            # Array is a validated tuple
            return tuple(getattr(items_schema[item_index], method_name)(instance[item_index])
                         for item_index in range(len(items_schema)))
        else:
            # Array is any unvalidated sequence
            return instance


class JsonObjectSchema(JsonSchema):

    def __init__(self,
                 properties: Mapping[str, JsonSchema] = None,
                 additional_properties: bool = None,
                 min_properties: int = None,
                 max_properties: int = None,
                 required: Sequence[str] = None,
                 **kwargs):
        super().__init__(type='object', **kwargs)
        self.properties = properties
        self.additional_properties = additional_properties
        self.min_properties = min_properties
        self.max_properties = max_properties
        self.required = required

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.properties is not None:
            d.update(properties={k: v.to_dict() for k, v in self.properties.items()})
        if self.additional_properties is not None:
            d.update(additionalProperties=self.additional_properties)
        if self.min_properties is not None:
            d.update(minProperties=self.min_properties)
        if self.max_properties is not None:
            d.update(maxProperties=self.max_properties)
        if self.required is not None:
            d.update(required=self.required)
        return d

    def to_json_instance(self, obj: Any) -> Mapping[str, Any]:
        if self.obj_to_json is not None:
            return self.obj_to_json(obj)
        return self._convert_instance(obj, 'to_json_instance')

    def from_json_instance(self, instance: Mapping[str, Any]) -> Any:
        deserialized_instance = self._convert_instance(instance, 'from_json_instance')
        return self.json_to_obj(**deserialized_instance) if self.json_to_obj is not None else deserialized_instance

    def _convert_instance(self, instance: Mapping[str, Any], method_name: str) -> Mapping[str, Any]:
        converted_instance = dict()

        if self.properties:
            for property_name, property_schema in self.properties.items():
                if property_name in instance:
                    property_value = instance[property_name]
                else:
                    property_value = property_schema.default
                converted_property_value = getattr(property_schema, method_name)(property_value)
                if converted_property_value is not UNDEFINED:
                    converted_instance[property_name] = converted_property_value

        # Note, additional_properties defaults to True
        if self.additional_properties is None or self.additional_properties:
            for property_name, property_value in instance.items():
                if property_name not in converted_instance and property_value is not UNDEFINED:
                    converted_instance[property_name] = property_value

        return converted_instance
