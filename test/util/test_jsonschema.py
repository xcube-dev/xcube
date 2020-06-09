import unittest
from collections import namedtuple

from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonIntegerSchema
from xcube.util.jsonschema import JsonNullSchema
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema


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

    def test_to_json_integer(self):
        self.assertEqual(45, JsonIntegerSchema().to_instance(45))

    def test_from_json_integer(self):
        self.assertEqual(45, JsonIntegerSchema().from_instance(45))


class JsonNumberSchemaTest(unittest.TestCase):
    def test_to_json_number(self):
        self.assertEqual(0.6, JsonNumberSchema().to_instance(0.6))

    def test_from_json_number(self):
        self.assertEqual(0.6, JsonNumberSchema().from_instance(0.6))


class JsonStringSchemaTest(unittest.TestCase):
    def test_to_json_string(self):
        self.assertEqual('pieps', JsonStringSchema().to_instance('pieps'))

    def test_from_json_string(self):
        self.assertEqual('pieps', JsonStringSchema().from_instance('pieps'))


class JsonArraySchemaTest(unittest.TestCase):

    def test_to_json_array(self):
        self.assertEqual([False, 2, 'U'], JsonArraySchema().to_instance([False, 2, 'U']))

    def test_from_json_array(self):
        self.assertEqual([False, 2, 'U'], JsonArraySchema().from_instance([False, 2, 'U']))

    def test_from_json_array_object(self):
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

    def test_process_kwargs(self):
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
