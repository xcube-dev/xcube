# The MIT License (MIT)
# Copyright (c) 2020 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Mapping, Sequence, Union, Tuple

import jsonschema

from xcube.util.ipython import register_json_formatter
from xcube.util.undefined import UNDEFINED

Factory = Callable[..., Any]
Serializer = Callable[..., Any]

_TYPES_ENUM = {'null', 'boolean', 'integer', 'number', 'string', 'array', 'object'}
_NUMERIC_TYPES_ENUM = {'integer', 'number'}


class JsonSchema(ABC):

    # noinspection PyShadowingBuiltins
    def __init__(self,
                 type: str,
                 default: Any = UNDEFINED,
                 const: Any = UNDEFINED,
                 enum: Sequence[Any] = None,
                 nullable: bool = None,
                 title: str = None,
                 description: str = None,
                 factory: Factory = None,
                 serializer: Serializer = None):
        if type not in _TYPES_ENUM:
            names = ', '.join(map(lambda t: f'"{t}"', sorted(list(_TYPES_ENUM))))
            raise ValueError(f'type must be one of {names}')
        if factory is not None and not callable(factory):
            raise ValueError('factory must be callable')
        if serializer is not None and not callable(serializer):
            raise ValueError('serializer must be callable')
        self.type = type
        self.default = default
        self.const = const
        self.enum = list(enum) if enum is not None else None
        self.nullable = nullable
        self.title = title
        self.description = description
        self.factory = factory
        self.serializer = serializer

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
        """Validate JSON value *instance*."""
        jsonschema.validate(instance=instance, schema=self.to_dict())

    def to_instance(self, value: Any) -> Any:
        """Convert Python object *value* into JSON value and return the validated result."""
        json_instance = self._to_unvalidated_instance(value)
        self.validate_instance(json_instance)
        return json_instance

    def from_instance(self, instance: Any) -> Any:
        """Validate JSON value *instance* and convert it into a Python object."""
        self.validate_instance(instance)
        return self._from_validated_instance(instance)

    @abstractmethod
    def _to_unvalidated_instance(self, value: Any) -> Any:
        """Turn Python object *value* into an unvalidated JSON value."""

    @abstractmethod
    def _from_validated_instance(self, instance: Any) -> Any:
        """Turn validated JSON value *instance* into a Python object."""


class JsonSimpleTypeSchema(JsonSchema, ABC):
    # noinspection PyShadowingBuiltins
    def __init__(self, type: str, **kwargs):
        super().__init__(type, **kwargs)

    def _to_unvalidated_instance(self, value: Any) -> Any:
        return self.serializer(value) if self.serializer is not None else value

    def _from_validated_instance(self, instance: Any) -> Any:
        return self.factory(instance) if self.factory is not None else instance


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
            names = ', '.join(sorted(map(lambda t: f'"{t}"', _NUMERIC_TYPES_ENUM)))
            raise ValueError(f'Type must be one of {names}')
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

    def _to_unvalidated_instance(self, value: Sequence[Any]) -> Sequence[Any]:
        if self.serializer:
            return self.serializer(value)
        return self._convert_sequence(value, '_to_unvalidated_instance')

    def _from_validated_instance(self, instance: Sequence[Any]) -> Any:
        obj = self._convert_sequence(instance, '_from_validated_instance')
        return self.factory(obj) if self.factory is not None else obj

    def _convert_sequence(self, sequence: Sequence[Any], method_name: str) -> Sequence[Any]:
        items_schema = self.items
        if isinstance(items_schema, JsonSchema):
            # Sequence turned into list with items_schema applying to all elements
            return [getattr(items_schema, method_name)(item)
                    for item in sequence]
        elif items_schema is not None:
            # Sequence turned into tuple with schema for every position
            return [getattr(items_schema[item_index], method_name)(sequence[item_index])
                    for item_index in range(len(items_schema))]
        # Sequence returned as-is, without schema
        return sequence


class JsonObjectSchema(JsonSchema):

    # TODO: also address property dependencies

    def __init__(self,
                 properties: Mapping[str, JsonSchema] = None,
                 additional_properties: Union[bool, JsonSchema] = None,
                 min_properties: int = None,
                 max_properties: int = None,
                 required: Sequence[str] = None,
                 dependencies: Mapping[str, Union[Sequence[str], JsonSchema]] = None,
                 **kwargs):
        super().__init__(type='object', **kwargs)
        self.properties = dict(properties) if properties else dict()
        self.additional_properties = additional_properties
        self.min_properties = min_properties
        self.max_properties = max_properties
        self.required = set(required) if required else set()
        self.dependencies = dict(dependencies) if dependencies else None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.properties:
            d.update(properties={k: v.to_dict() for k, v in self.properties.items()})
        if self.additional_properties is not None:
            d.update(additionalProperties=self.additional_properties.to_dict() \
                if isinstance(self.additional_properties, JsonSchema) else self.additional_properties)
        if self.min_properties is not None:
            d.update(minProperties=self.min_properties)
        if self.max_properties is not None:
            d.update(maxProperties=self.max_properties)
        if self.required:
            d.update(required=list(self.required))
        if self.dependencies:
            d.update(dependencies={k: (v.to_dict() if isinstance(v, JsonSchema) else v)
                                   for k, v in self.dependencies.items()})
        return d

    # TODO: move away. this is a special-purpose utility
    def process_kwargs_subset(self,
                              kwargs: Dict[str, Any],
                              keywords: Sequence[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Utility that helps processing keyword-arguments. in *kwargs*:

        Pop every keyword in *keywords* contained in this object schema's properties
        from *kwargs* and put the keyword and value from *kwargs* into a new dictionary.

        Return a 2-element tuple, first *kwargs* with, second the and *kwargs* without the keywords from *keywords*.

        The original *kwargs* is not touched.

        :return a tuple of two new keyword-argument dictionaries
        """
        old_kwargs = dict(kwargs)
        new_kwargs = {}
        for k in keywords:
            if k in old_kwargs:
                new_kwargs[k] = old_kwargs.pop(k)
            elif k in self.properties:
                property_schema = self.properties[k]
                if property_schema.const is not UNDEFINED:
                    new_kwargs[k] = property_schema.const
                elif property_schema.default is not UNDEFINED and k in self.required:
                    if property_schema.default is not UNDEFINED:
                        new_kwargs[k] = property_schema.default
        return new_kwargs, old_kwargs

    def _to_unvalidated_instance(self, value: Any) -> Mapping[str, Any]:
        if self.serializer is not None:
            return self.serializer(value)
        return self._convert_mapping(value, '_to_unvalidated_instance')

    def _from_validated_instance(self, instance: Mapping[str, Any]) -> Any:
        obj = self._convert_mapping(instance, '_from_validated_instance')
        return self.factory(**obj) if self.factory is not None else obj

    def _convert_mapping(self, mapping: Mapping[str, Any], method_name: str) -> Mapping[str, Any]:
        # TODO: recognise self.dependencies. if dependency is again a schema, compute effective schema
        #       by merging this and the dependency.

        converted_mapping = dict()

        required_set = set(self.required) if self.required else set()

        for property_name, property_schema in self.properties.items():
            if property_name in mapping:
                property_value = mapping[property_name]
                converted_property_value = getattr(property_schema, method_name)(property_value)
                converted_mapping[property_name] = converted_property_value
            else:
                property_value = property_schema.default
                if property_value is not UNDEFINED:
                    converted_property_value = getattr(property_schema, method_name)(property_value)
                    if property_name in required_set or converted_property_value is not None:
                        converted_mapping[property_name] = converted_property_value

        # Note, additional_properties defaults to True
        if self.additional_properties is None or self.additional_properties:
            for property_name, property_value in mapping.items():
                if property_name not in converted_mapping and property_value is not UNDEFINED:
                    converted_mapping[property_name] = property_value

        return converted_mapping


register_json_formatter(JsonSchema)
