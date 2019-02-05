import unittest

from xcube.objreg import ObjRegistry, get_obj_registry, ObjRegistration


class A:
    pass


class B(A):
    pass


class ObjRegistryTest(unittest.TestCase):
    def test_get_obj_registry(self):
        self.assertIsInstance(get_obj_registry(), ObjRegistry)

    def test_get(self):
        obj_registry = ObjRegistry()
        with self.assertRaises(KeyError):
            obj_registry.get('test')

    def test_put_illegal(self):
        obj_registry = ObjRegistry()
        with self.assertRaises(ValueError) as cm:
            obj_registry.put('test', A(), type=B)
        self.assertEqual("obj must be an instance of <class 'test.test_objreg.B'>", f'{cm.exception}')

    def test_put_has_get(self):
        obj_registry = ObjRegistry()
        obj = A()
        obj_registration = obj_registry.put('test', obj)
        self.assertEqual(True, obj_registry.has('test'))
        self.assertIs(obj, obj_registry.get('test'))
        self.assertIsInstance(obj_registration, ObjRegistration)
        self.assertEqual('test', obj_registration.name)
        self.assertEqual(object, obj_registration.type)
        self.assertEqual(obj, obj_registration.obj)
        self.assertEqual(False, obj_registration.deleted)
        obj_registration.delete()
        self.assertEqual(True, obj_registration.deleted)
        self.assertEqual(False, obj_registry.has('test'))

    def test_put_multiple(self):
        obj_registry = ObjRegistry()
        obj1 = A()
        obj2 = A()
        obj3 = A()
        obj4 = B()
        obj5 = B()
        obj6 = B()
        obj_registry.put('a1', obj1, type=A)
        obj_registry.put('a2', obj2, type=A)
        obj_registry.put('a3', obj3, type=A)
        obj_registry.put('b1', obj4, type=B)
        obj_registry.put('b2', obj5, type=B)
        obj_registry.put('b3', obj6, type=B)
        self.assertIs(obj1, obj_registry.get('a1', type=A))
        self.assertIs(obj2, obj_registry.get('a2', type=A))
        self.assertIs(obj3, obj_registry.get('a3', type=A))
        self.assertIs(obj4, obj_registry.get('b1', type=B))
        self.assertIs(obj5, obj_registry.get('b2', type=B))
        self.assertIs(obj6, obj_registry.get('b3', type=B))
