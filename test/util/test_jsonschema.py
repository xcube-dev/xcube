import unittest
from collections import namedtuple
from typing import Dict, Any

from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonComplexSchema
from xcube.util.jsonschema import JsonDateSchema
from xcube.util.jsonschema import JsonDatetimeSchema
from xcube.util.jsonschema import JsonIntegerSchema
from xcube.util.jsonschema import JsonNullSchema
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonObject
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonSimpleSchema
from xcube.util.jsonschema import JsonStringSchema


class JsonComplexSchemaTest(unittest.TestCase):

    def test_base_props_validated(self):
        with self.assertRaises(ValueError) as cm:
            JsonComplexSchema()
        self.assertEqual('exactly one of one_of, any_of, all_of must be given',
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            JsonComplexSchema(one_of=[JsonStringSchema(), JsonIntegerSchema()],
                              all_of=[JsonStringSchema(), JsonIntegerSchema()])
        self.assertEqual('exactly one of one_of, any_of, all_of must be given',
                         f'{cm.exception}')

    def test_to_dict(self):
        self.assertEqual(
            {'oneOf': [{'multipleOf': 5, 'type': 'integer'},
                       {'multipleOf': 3, 'type': 'integer'}]},
            JsonComplexSchema(one_of=[JsonIntegerSchema(multiple_of=5),
                                      JsonIntegerSchema(multiple_of=3)]).to_dict())
        self.assertEqual(
            {'anyOf': [{'multipleOf': 5, 'type': 'integer'},
                       {'multipleOf': 3, 'type': 'integer'}]},
            JsonComplexSchema(any_of=[JsonIntegerSchema(multiple_of=5),
                                      JsonIntegerSchema(multiple_of=3)]).to_dict())
        self.assertEqual(
            {'allOf': [{'multipleOf': 5, 'type': 'integer'},
                       {'multipleOf': 3, 'type': 'integer'}]},
            JsonComplexSchema(all_of=[JsonIntegerSchema(multiple_of=5),
                                      JsonIntegerSchema(multiple_of=3)]).to_dict())


class JsonSimpleSchemaTest(unittest.TestCase):

    def test_base_props_validated(self):
        with self.assertRaises(ValueError) as cm:
            JsonSimpleSchema('int')
        self.assertEqual('type must be one of "array", "boolean", "integer", "null", "number", "object", "string"',
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            JsonSimpleSchema('integer', factory='int')
        self.assertEqual('factory must be callable',
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            JsonSimpleSchema('integer', serializer='int')
        self.assertEqual('serializer must be callable',
                         f'{cm.exception}')

    def test_to_dict(self):
        self.assertEqual(
            {
                'type': ['integer', 'null'],
                'title': 'Number of Gnarzes',
                'description': 'Not really required',
                'enum': [10, 100, 1000],
                'default': 10,
                'const': 100,
            },
            JsonSimpleSchema('integer',
                             title='Number of Gnarzes',
                             description='Not really required',
                             enum=[10, 100, 1000],
                             const=100,
                             default=10,
                             nullable=True).to_dict())


class JsonNullSchemaTest(unittest.TestCase):

    def test_to_json_null(self):
        self.assertEqual(None, JsonNullSchema().to_instance(None))

    def test_from_json_null(self):
        self.assertEqual(None, JsonNullSchema().from_instance(None))


class JsonBooleanSchemaTest(unittest.TestCase):

    def test_to_json_boolean(self):
        self.assertEqual(True, JsonBooleanSchema().to_instance(True))

    def test_from_json_boolean(self):
        self.assertEqual(True, JsonBooleanSchema().from_instance(True))


class JsonIntegerSchemaTest(unittest.TestCase):

    def test_to_instance(self):
        self.assertEqual(45, JsonIntegerSchema(minimum=0,
                                               maximum=100,
                                               multiple_of=5).to_instance(45))

    def test_from_instance(self):
        self.assertEqual(45, JsonIntegerSchema(minimum=0,
                                               maximum=100,
                                               multiple_of=5).from_instance(45))


class JsonNumberSchemaTest(unittest.TestCase):
    def test_to_instance(self):
        self.assertEqual(0.6, JsonNumberSchema(exclusive_minimum=0,
                                               exclusive_maximum=1).to_instance(0.6))

    def test_from_instance(self):
        self.assertEqual(6, JsonNumberSchema(minimum=0,
                                             maximum=10,
                                             multiple_of=2).from_instance(6))

    def test_check_type(self):
        with self.assertRaises(ValueError) as cm:
            JsonNumberSchema(type='float')
        self.assertEqual('Type must be one of "integer", "number"', f'{cm.exception}')

    def test_to_dict(self):
        self.assertEqual({'type': 'number'}, JsonNumberSchema().to_dict())
        self.assertEqual({'type': 'number'}, JsonNumberSchema(nullable=False).to_dict())
        self.assertEqual({'type': ['number', 'null']}, JsonNumberSchema(nullable=True).to_dict())
        self.assertEqual({'type': 'number',
                          'exclusiveMinimum': 0,
                          'maximum': 100,
                          'multipleOf': 10},
                         JsonNumberSchema(exclusive_minimum=0,
                                          maximum=100,
                                          multiple_of=10).to_dict())
        self.assertEqual({'type': 'integer',
                          'minimum': 100,
                          'exclusiveMaximum': 200,
                          'multipleOf': 20},
                         JsonIntegerSchema(minimum=100,
                                           exclusive_maximum=200,
                                           multiple_of=20).to_dict())


class JsonStringSchemaTest(unittest.TestCase):
    def test_to_instance(self):
        self.assertEqual('pieps', JsonStringSchema().to_instance('pieps'))
        self.assertEqual('pieps', JsonStringSchema(min_length=0,
                                                   max_length=10).to_instance('pieps'))
        self.assertEqual('pieps', JsonStringSchema(pattern='.*').to_instance('pieps'))
        self.assertEqual('2020-01-03T03:30:01.99+03:30',
                         JsonStringSchema(format='date-time').
                         to_instance('2020-01-03T03:30:01.99+03:30'))

    def test_from_instance(self):
        self.assertEqual('pieps', JsonStringSchema().from_instance('pieps'))

    def test_to_dict(self):
        self.assertEqual({'type': 'string'}, JsonStringSchema().to_dict())
        self.assertEqual({'type': 'string'}, JsonStringSchema(nullable=False).to_dict())
        self.assertEqual({'type': ['string', 'null']}, JsonStringSchema(nullable=True).to_dict())
        self.assertEqual({'type': 'string', 'format': 'uri'}, JsonStringSchema(format='uri').to_dict())


class JsonDateSchemaTest(unittest.TestCase):

    def test_to_instance(self):
        self.assertEqual('2020-06-03',
                         JsonDateSchema(min_date='2020-02-01',
                                        max_date='2020-07-05').to_instance('2020-06-03'))

    def test_to_dict(self):
        self.assertEqual({'type': 'string',
                          'format': 'date',
                          'minDate': '2020-02-01',
                          'maxDate': '2020-07-05'
                          },
                         JsonDateSchema(min_date='2020-02-01',
                                        max_date='2020-07-05').to_dict())

        self.assertEqual({'type': ['string', 'null'],
                          'format': 'date',
                          'minDate': '2020-02-01',
                          'maxDate': '2020-07-05'
                          },
                         JsonDateSchema(min_date='2020-02-01',
                                        max_date='2020-07-05',
                                        nullable=True).to_dict())

    def test_store_date_limits(self):
        minimum = '1981-05-06'
        maximum = '1982-09-15'
        schema = JsonDateSchema(min_date=minimum,
                                max_date=maximum)
        self.assertEqual(minimum, schema.min_date)
        self.assertEqual(maximum, schema.max_date)

    def test_min_max_validity_checks(self):
        with self.assertRaises(ValueError):
            JsonDateSchema(min_date='2002-02-02T10:20:14')
        with self.assertRaises(ValueError):
            JsonDateSchema(max_date='pippo')

    def test_new_range(self):
        self.assertEqual(
            {'type': 'array',
             'items': [{'type': 'string',
                        'format': 'date',
                        },
                       {'type': 'string',
                        'format': 'date',
                        }
                       ],
             },
            JsonDateSchema.new_range().to_dict())

        self.assertEqual(
            {'type': ['array', 'null'],
             'items': [{'type': ['string', 'null'],
                        'format': 'date',
                        'minDate': '2020-02-01',
                        'maxDate': '2020-07-05',
                        },
                       {'type': ['string', 'null'],
                        'format': 'date',
                        'minDate': '2020-02-01',
                        'maxDate': '2020-07-05',
                        }
                       ],
             },
            JsonDateSchema.new_range(min_date='2020-02-01',
                                     max_date='2020-07-05',
                                     nullable=True).to_dict())


class JsonDatetimeSchemaTest(unittest.TestCase):

    def test_to_instance(self):
        self.assertEqual('2020-06-12T12:30:19Z',
                         JsonDatetimeSchema(min_datetime='2020-02-01T00:00:00Z',
                                            max_datetime='2020-07-05T00:00:00Z').to_instance(
                             '2020-06-12T12:30:19Z'))

    def test_to_dict(self):
        self.assertEqual({'type': 'string',
                          'format': 'date-time',
                          'minDatetime': '2020-02-01T00:00:00Z',
                          'maxDatetime': '2020-07-05T00:00:00Z'},
                         JsonDatetimeSchema(min_datetime='2020-02-01T00:00:00Z',
                                            max_datetime='2020-07-05T00:00:00Z').to_dict())

    def test_store_date_limits(self):
        minimum = '1981-05-06T00:00:00+00:00'
        maximum = '1982-09-15T00:00:00+00:00'
        schema = JsonDatetimeSchema(min_datetime=minimum,
                                    max_datetime=maximum)
        self.assertEqual(minimum, schema.min_datetime)
        self.assertEqual(maximum, schema.max_datetime)

    def test_min_max_validity_checks(self):
        with self.assertRaises(ValueError):
            JsonDatetimeSchema(  # missing timezone specifier
                min_datetime='1980-02-03T12:34:56',
                max_datetime='1982-02-03T23:34:56+05:00')
        with self.assertRaises(ValueError):
            JsonDatetimeSchema(min_datetime='1980-02-03T12:34:56-08:00',
                               # invalid date
                               max_datetime='1985-01-32')

    def test_new_range(self):
        self.assertEqual(
            {'type': 'array',
             'items': [{'type': 'string',
                        'format': 'date-time',
                        },
                       {'type': 'string',
                        'format': 'date-time',
                        }
                       ],
             },
            JsonDatetimeSchema.new_range().to_dict())

        self.assertEqual(
            {'type': ['array', 'null'],
             'items': [{'type': ['string', 'null'],
                        'format': 'date-time',
                        'minDatetime': '2002-01-01T00:00:00Z',
                        'maxDatetime': '2020-01-01T00:00:00Z',
                        },
                       {'type': ['string', 'null'],
                        'format': 'date-time',
                        'minDatetime': '2002-01-01T00:00:00Z',
                        'maxDatetime': '2020-01-01T00:00:00Z',
                        }
                       ],
             },
            JsonDatetimeSchema.new_range(min_datetime='2002-01-01T00:00:00Z',
                                         max_datetime='2020-01-01T00:00:00Z',
                                         nullable=True).to_dict())


class JsonArraySchemaTest(unittest.TestCase):

    def test_to_instance(self):
        self.assertEqual([False, 2, 'U'],
                         JsonArraySchema().to_instance([False, 2, 'U']))

    def test_to_instance_tuple(self):
        self.assertEqual([False, 2, 'U'],
                         JsonArraySchema(items=[JsonBooleanSchema(),
                                                JsonIntegerSchema(),
                                                JsonStringSchema()]).to_instance([False, 2, 'U']))

    def test_from_instance(self):
        self.assertEqual([False, 2, 'U'],
                         JsonArraySchema().from_instance([False, 2, 'U']))

    def test_from_instance_tuple(self):
        self.assertEqual([False, 2, 'U'],
                         JsonArraySchema(items=[JsonBooleanSchema(),
                                                JsonIntegerSchema(),
                                                JsonStringSchema()]).from_instance([False, 2, 'U']))

    def test_tuple_validates_as_array(self):
        self.assertTupleEqual((1, 2, 3),
                              JsonArraySchema().to_instance((1, 2, 3)))

    def test_from_instance_array_object(self):
        value = [{'name': 'Bibo', 'age': 15},
                 {'name': 'Ernie', 'age': 12}]

        person_schema = JsonObjectSchema(
            properties=dict(name=JsonStringSchema(), age=JsonIntegerSchema(),
                            deleted=JsonBooleanSchema(default=False)))
        schema = JsonArraySchema(items=person_schema)
        self.assertEqual([{'name': 'Bibo', 'age': 15, 'deleted': False},
                          {'name': 'Ernie', 'age': 12, 'deleted': False}],
                         schema.from_instance(value))

        Person = namedtuple('Person', ['name', 'age', 'deleted'])
        person_schema.factory = Person
        self.assertEqual([Person(name='Bibo', age=15, deleted=False),
                          Person(name='Ernie', age=12, deleted=False)],
                         schema.from_instance(value))

        Assignment = namedtuple('Assignment', ['persons'])

        def factory(persons):
            return Assignment(persons=persons)

        schema.factory = factory
        self.assertEqual(Assignment(persons=[Person(name='Bibo', age=15, deleted=False),
                                             Person(name='Ernie', age=12, deleted=False)]),
                         schema.from_instance(value))


class JsonObjectSchemaTest(unittest.TestCase):

    def test_from_json_object(self):
        value = {'name': 'Bibo', 'age': 12, 'deleted': True}

        person_schema = JsonObjectSchema(properties=dict(name=JsonStringSchema(),
                                                         age=JsonIntegerSchema(),
                                                         deleted=JsonBooleanSchema(default=False)))
        self.assertEqual(value,
                         person_schema.from_instance(value))

        self.assertEqual({'name': 'Bibo', 'age': 12, 'deleted': False},
                         person_schema.from_instance({'name': 'Bibo', 'age': 12}))

        Person = namedtuple('Person', ['name', 'age', 'deleted'])
        person_schema.factory = Person
        self.assertEqual(Person(name='Bibo', age=12, deleted=True),
                         person_schema.from_instance(value))

    def test_to_json_object(self):
        person_schema = JsonObjectSchema(properties=dict(name=JsonStringSchema(),
                                                         age=JsonIntegerSchema(),
                                                         deleted=JsonBooleanSchema(default=False)))

        value = {'name': 'Bibo', 'age': 12, 'deleted': True}

        self.assertEqual(value,
                         person_schema.to_instance(value))

        # ok, because person_schema does not explicitly say additional_properties=False
        value_extra = {'name': 'Bibo', 'age': 12, 'deleted': True, 'comment': 'Hello!'}
        self.assertEqual(value_extra,
                         person_schema.to_instance(value_extra))

        Person = namedtuple('Person', ['name', 'age', 'deleted'])

        def serialize(person: Person) -> Dict[str, Any]:
            return person._asdict()

        person_schema.serializer = serialize

        person = Person(**value)
        self.assertEqual(value,
                         person_schema.to_instance(person))

    def test_from_json_object_object(self):
        person_schema = JsonObjectSchema(properties=dict(name=JsonStringSchema(),
                                                         age=JsonIntegerSchema(),
                                                         deleted=JsonBooleanSchema(default=False)))
        schema = JsonObjectSchema(properties=dict(person=person_schema))

        value = {'person': {'name': 'Bibo', 'age': 15}}

        self.assertEqual({'person': {'name': 'Bibo', 'age': 15, 'deleted': False}},
                         schema.from_instance(value))

        Assignment = namedtuple('Assignment', ['person'])
        schema.factory = Assignment
        self.assertEqual(Assignment(person={'name': 'Bibo', 'age': 15, 'deleted': False}),
                         schema.from_instance(value))

        Person = namedtuple('Person', ['name', 'age', 'deleted'])
        person_schema.factory = Person
        self.assertEqual(Assignment(person=Person(name='Bibo', age=15, deleted=False)),
                         schema.from_instance(value))

    def test_from_json_object_array_object(self):
        person_schema = JsonObjectSchema(properties=dict(name=JsonStringSchema(),
                                                         age=JsonIntegerSchema(),
                                                         deleted=JsonBooleanSchema(default=False)))

        schema = JsonObjectSchema(properties=dict(persons=JsonArraySchema(items=person_schema)))

        value = {'persons': [{'name': 'Bibo', 'age': 15},
                             {'name': 'Ernie', 'age': 12}]}

        self.assertEqual({'persons': [{'name': 'Bibo', 'age': 15, 'deleted': False},
                                      {'name': 'Ernie', 'age': 12, 'deleted': False}]},
                         schema.from_instance(value))

        Assignment = namedtuple('Assignment', ['persons'])
        schema.factory = Assignment
        self.assertEqual(Assignment(persons=[{'name': 'Bibo', 'age': 15, 'deleted': False},
                                             {'name': 'Ernie', 'age': 12, 'deleted': False}]),
                         schema.from_instance(value))

        Person = namedtuple('Person', ['name', 'age', 'deleted'])
        person_schema.factory = Person
        self.assertEqual(Assignment(persons=[Person(name='Bibo', age=15, deleted=False),
                                             Person(name='Ernie', age=12, deleted=False)]),
                         schema.from_instance(value))

    def test_from_json_object_additional_properties_is_schema(self):
        Person = namedtuple('Person', ['name', 'age', 'deleted'])

        person_schema = JsonObjectSchema(
            properties=dict(
                name=JsonStringSchema(),
                age=JsonIntegerSchema(),
                deleted=JsonBooleanSchema(default=False)
            ),
            factory=Person,
        )

        schema = JsonObjectSchema(
            additional_properties=person_schema,
        )

        value = {
            'p1': {'name': 'Bibo', 'age': 15, 'deleted': True},
            'p2': {'name': 'Ernie', 'age': 12, 'deleted': False},
        }

        self.assertEqual(
            {
                'p1': Person(name='Bibo', age=15, deleted=True),
                'p2': Person(name='Ernie', age=12, deleted=False),
            },
            schema.from_instance(value))

    def test_process_kwargs_subset(self):
        schema = JsonObjectSchema(
            properties=dict(
                client_id=JsonStringSchema(default='bibo'),
                client_secret=JsonStringSchema(default='2w908435t'),
                geom=JsonStringSchema(),
                crs=JsonStringSchema(const='WGS84'),
                spatial_res=JsonNumberSchema(),
                time_range=JsonStringSchema(),
                time_period=JsonStringSchema(default='8D'),
                max_cache_size=JsonIntegerSchema(),
            ),
            required=['client_id', 'client_secret', 'geom', 'crs', 'spatial_res', 'time_range'],
        )

        kwargs = dict(client_secret='094529g',
                      geom='POINT (12.2, 53.9)',
                      spatial_res=0.5,
                      time_range='2010,2014',
                      max_cache_size=2 ** 32)

        cred_kwargs, kwargs = schema.process_kwargs_subset(kwargs, ['client_id', 'client_secret'])
        self.assertEqual(dict(client_id='bibo', client_secret='094529g'),
                         cred_kwargs)
        self.assertEqual(dict(geom='POINT (12.2, 53.9)',
                              spatial_res=0.5,
                              time_range='2010,2014',
                              max_cache_size=2 ** 32),
                         kwargs)

        ds_kwargs, kwargs = schema.process_kwargs_subset(kwargs, ['geom', 'crs',
                                                                  'spatial_res', 'time_range', 'time_period'])
        self.assertEqual(dict(crs='WGS84',
                              geom='POINT (12.2, 53.9)',
                              spatial_res=0.5,
                              time_range='2010,2014'),
                         ds_kwargs)
        self.assertEqual(dict(max_cache_size=2 ** 32),
                         kwargs)

    def test_to_dict(self):
        schema = JsonObjectSchema(
            properties=dict(
                consolidated=JsonBooleanSchema()
            )
        )
        self.assertEqual(
            {
                'type': 'object',
                'properties': {
                    'consolidated': {
                        'type': 'boolean'
                    }
                },
            },
            schema.to_dict()
        )


class IHasAProperty(JsonObject):
    def __init__(self, name, age):
        # A private state property
        self._name = name
        # A state property
        self.age = age

    @property
    def id(self):
        # A derived property
        return f'{self.name}_{self.age}'

    @property
    def name(self):
        # A state property
        return self._name

    @classmethod
    def get_schema(cls):
        return JsonObjectSchema(
            properties=dict(
                name=JsonStringSchema(),
                age=JsonIntegerSchema()
            ),
            required=['name', 'age'],
            factory=cls
        )


class IHasAProperty2(IHasAProperty):
    @classmethod
    def get_schema(cls):
        schema = super().get_schema()
        schema.additional_properties = False
        return schema


class JsonObjectWithPropertyTest(unittest.TestCase):
    def test_with_additional_properties(self):
        obj_dict = {'name': 'bibo', 'age': 2021 - 1969}
        obj = IHasAProperty.from_dict(obj_dict)
        actual_dict = obj.to_dict()
        self.assertEqual({'age': 52, 'id': 'bibo_52', 'name': 'bibo'}, actual_dict)

    def test_without_additional_properties(self):
        obj_dict = {'name': 'bibo', 'age': 2021 - 1969}
        obj = IHasAProperty2.from_dict(obj_dict)
        actual_dict = obj.to_dict()
        self.assertEqual(obj_dict, actual_dict)
