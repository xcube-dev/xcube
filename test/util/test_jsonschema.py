import unittest
from collections import namedtuple
from typing import Dict, Any

from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonIntegerSchema
from xcube.util.jsonschema import JsonNullSchema
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonSchema
from xcube.util.jsonschema import JsonSimpleTypeSchema
from xcube.util.jsonschema import JsonStringSchema


class JsonSchemaTest(unittest.TestCase):

    def test_base_props_validated(self):
        class MyIntSchema(JsonSimpleTypeSchema, JsonSchema):
            def __init__(self, type: str, **kwargs):
                super().__init__(type=type, **kwargs)

        with self.assertRaises(ValueError) as cm:
            MyIntSchema('int')
        self.assertEqual('type must be one of "array", "boolean", "integer", "null", "number", "object", "string"',
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            MyIntSchema('integer', factory='int')
        self.assertEqual('factory must be callable',
                         f'{cm.exception}')

        with self.assertRaises(ValueError) as cm:
            MyIntSchema('integer', serializer='int')
        self.assertEqual('serializer must be callable',
                         f'{cm.exception}')

    def test_to_dict(self):
        class MyIntSchema(JsonSimpleTypeSchema, JsonSchema):
            def __init__(self, type: str, **kwargs):
                super().__init__(type=type, **kwargs)

        self.assertEqual(
            {
                'type': ['integer', 'null'],
                'title': 'Number of Gnarzes',
                'description': 'Not really required',
                'enum': [10, 100, 1000],
                'default': 10,
                'const': 100,
            },
            MyIntSchema('integer',
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


class JsonStringSchemaTest(unittest.TestCase):
    def test_to_instance(self):
        self.assertEqual('pieps', JsonStringSchema().to_instance('pieps'))
        self.assertEqual('pieps', JsonStringSchema(min_length=0,
                                                   max_length=10).to_instance('pieps'))
        self.assertEqual('pieps', JsonStringSchema(pattern='.*').to_instance('pieps'))
        self.assertEqual('2020-01-03', JsonStringSchema(format='date').to_instance('2020-01-03'))
        self.assertEqual('2020-06-03',
                         JsonStringSchema(format='datetime',
                                          min_datetime='2020-02-01',
                                          max_datetime='2020-07-05').
                         to_instance('2020-06-03'))

    def test_from_instance(self):
        self.assertEqual('pieps', JsonStringSchema().from_instance('pieps'))

    def test_store_date_limits(self):
        minimum = '1981-05-06'
        maximum = '1982-09-15'
        schema = JsonStringSchema(format='datetime', min_datetime=minimum,
                                  max_datetime=maximum)
        self.assertEqual(minimum, schema.min_datetime)
        self.assertEqual(maximum, schema.max_datetime)


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
