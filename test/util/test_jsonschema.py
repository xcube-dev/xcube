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
        self.assertEqual(None, JsonNullSchema().to_json_instance(None))

    def test_from_json_null(self):
        self.assertEqual(None, JsonNullSchema().from_json_instance(None))


class JsonBooleanSchemaTest(unittest.TestCase):

    def test_to_json_boolean(self):
        self.assertEqual(True, JsonBooleanSchema().to_json_instance(True))

    def test_from_json_boolean(self):
        self.assertEqual(True, JsonBooleanSchema().from_json_instance(True))


class JsonIntegerSchemaTest(unittest.TestCase):

    def test_to_json_integer(self):
        self.assertEqual(45, JsonIntegerSchema().to_json_instance(45))
    def test_from_json_integer(self):
        self.assertEqual(45, JsonIntegerSchema().from_json_instance(45))


class JsonNumberSchemaTest(unittest.TestCase):
    def test_to_json_number(self):
        self.assertEqual(0.6, JsonNumberSchema().to_json_instance(0.6))

    def test_from_json_number(self):
        self.assertEqual(0.6, JsonNumberSchema().from_json_instance(0.6))


class JsonStringSchemaTest(unittest.TestCase):
    def test_to_json_string(self):
        self.assertEqual('pieps', JsonStringSchema().to_json_instance('pieps'))

    def test_from_json_string(self):
        self.assertEqual('pieps', JsonStringSchema().from_json_instance('pieps'))


class JsonArraySchemaTest(unittest.TestCase):

    def test_to_json_array(self):
        self.assertEqual([False, 2, 'U'], JsonArraySchema().to_json_instance([False, 2, 'U']))

    def test_from_json_array(self):
        self.assertEqual([False, 2, 'U'], JsonArraySchema().from_json_instance([False, 2, 'U']))

    def test_from_json_array_object(self):
        value = [{'name': 'Bibo', 'age': 15},
                 {'name': 'Ernie', 'age': 12}]

        person_schema = JsonObjectSchema(
            properties=dict(name=JsonStringSchema(), age=JsonIntegerSchema(),
                            deleted=JsonBooleanSchema(default=False)))
        schema = JsonArraySchema(items=person_schema)
        self.assertEqual([{'name': 'Bibo', 'age': 15, 'deleted': False},
                          {'name': 'Ernie', 'age': 12, 'deleted': False}],
                         schema.from_json_instance(value))

        Person = namedtuple('Person', ['name', 'age', 'deleted'])
        person_schema.json_to_obj = Person
        self.assertEqual([Person(name='Bibo', age=15, deleted=False),
                          Person(name='Ernie', age=12, deleted=False)],
                         schema.from_json_instance(value))

        Assignment = namedtuple('Assignment', ['persons'])

        def factory(persons):
            return Assignment(persons=persons)

        schema.json_to_obj = factory
        self.assertEqual(Assignment(persons=[Person(name='Bibo', age=15, deleted=False),
                                             Person(name='Ernie', age=12, deleted=False)]),
                         schema.from_json_instance(value))


class JsonObjectSchemaTest(unittest.TestCase):

    def test_from_json_object(self):
        value = {'name': 'Bibo', 'age': 12, 'deleted': True}

        person_schema = JsonObjectSchema(properties=dict(name=JsonStringSchema(),
                                                         age=JsonIntegerSchema(),
                                                         deleted=JsonBooleanSchema(default=False)))
        self.assertEqual(value,
                         person_schema.from_json_instance(value))

        self.assertEqual({'name': 'Bibo', 'age': 12, 'deleted': False},
                         person_schema.from_json_instance({'name': 'Bibo', 'age': 12}))

        Person = namedtuple('Person', ['name', 'age', 'deleted'])
        person_schema.json_to_obj = Person
        self.assertEqual(Person(name='Bibo', age=12, deleted=True),
                         person_schema.from_json_instance(value))

    def test_from_json_object_object(self):
        person_schema = JsonObjectSchema(properties=dict(name=JsonStringSchema(),
                                                         age=JsonIntegerSchema(),
                                                         deleted=JsonBooleanSchema(default=False)))
        schema = JsonObjectSchema(properties=dict(person=person_schema))

        value = {'person': {'name': 'Bibo', 'age': 15}}

        self.assertEqual({'person': {'name': 'Bibo', 'age': 15, 'deleted': False}},
                         schema.from_json_instance(value))

        Assignment = namedtuple('Assignment', ['person'])
        schema.json_to_obj = Assignment
        self.assertEqual(Assignment(person={'name': 'Bibo', 'age': 15, 'deleted': False}),
                         schema.from_json_instance(value))

        Person = namedtuple('Person', ['name', 'age', 'deleted'])
        person_schema.json_to_obj = Person
        self.assertEqual(Assignment(person=Person(name='Bibo', age=15, deleted=False)),
                         schema.from_json_instance(value))

    def test_from_json_object_array_object(self):
        person_schema = JsonObjectSchema(properties=dict(name=JsonStringSchema(),
                                                         age=JsonIntegerSchema(),
                                                         deleted=JsonBooleanSchema(default=False)))

        schema = JsonObjectSchema(properties=dict(persons=JsonArraySchema(items=person_schema)))

        value = {'persons': [{'name': 'Bibo', 'age': 15},
                             {'name': 'Ernie', 'age': 12}]}

        self.assertEqual({'persons': [{'name': 'Bibo', 'age': 15, 'deleted': False},
                                      {'name': 'Ernie', 'age': 12, 'deleted': False}]},
                         schema.from_json_instance(value))

        Assignment = namedtuple('Assignment', ['persons'])
        schema.json_to_obj = Assignment
        self.assertEqual(Assignment(persons=[{'name': 'Bibo', 'age': 15, 'deleted': False},
                                             {'name': 'Ernie', 'age': 12, 'deleted': False}]),
                         schema.from_json_instance(value))

        Person = namedtuple('Person', ['name', 'age', 'deleted'])
        person_schema.json_to_obj = Person
        self.assertEqual(Assignment(persons=[Person(name='Bibo', age=15, deleted=False),
                                             Person(name='Ernie', age=12, deleted=False)]),
                         schema.from_json_instance(value))
