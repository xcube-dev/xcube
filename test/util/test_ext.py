import unittest

from xcube.util.ext import ExtensionRegistry, Extension, get_ext_registry


class A:
    pass


class B(A):
    pass


class ObjRegistryTest(unittest.TestCase):
    def test_get_ext_reg(self):
        self.assertIsInstance(get_ext_registry(), ExtensionRegistry)

    def test_get_ext_illegal(self):
        ext_reg = ExtensionRegistry()
        with self.assertRaises(KeyError):
            ext_reg.get_ext('A', 'test')

    def test_get_ext_obj_illegal(self):
        ext_reg = ExtensionRegistry()
        with self.assertRaises(KeyError):
            ext_reg.get_ext_obj('A', 'test')

    def test_add_has_get_del_ext(self):
        ext_reg = ExtensionRegistry()

        a_obj = A()

        class BFactory:
            # noinspection PyMethodMayBeStatic
            def load(self):
                return B()

        a_ext = ext_reg.add_ext('A', 'test', a_obj)
        b_ext = ext_reg.add_ext('B', 'test', BFactory())

        self.assertEqual(True, ext_reg.has_ext('A', 'test'))
        self.assertIsInstance(a_ext, Extension)
        self.assertEqual('test', a_ext.name)
        self.assertEqual('A', a_ext.type)
        self.assertEqual(False, a_ext.deleted)
        self.assertIs(a_obj, a_ext.obj)
        self.assertIs(a_obj, a_ext.obj)
        self.assertIs(a_obj, ext_reg.get_ext_obj('A', 'test'))
        self.assertIs(a_ext, ext_reg.get_ext('A', 'test'))

        self.assertEqual(True, ext_reg.has_ext('B', 'test'))
        self.assertIsInstance(b_ext, Extension)
        self.assertEqual('test', b_ext.name)
        self.assertEqual('B', b_ext.type)
        self.assertEqual(False, b_ext.deleted)
        b_obj = b_ext.obj
        self.assertIs(b_obj, b_ext.obj)
        self.assertIs(b_obj, ext_reg.get_ext_obj('B', 'test'))
        self.assertIs(b_ext, ext_reg.get_ext('B', 'test'))

        self.assertEqual([a_obj], ext_reg.get_all_ext_obj('A'))
        self.assertEqual([b_obj], ext_reg.get_all_ext_obj('B'))

        a_ext.delete()
        self.assertEqual(True, a_ext.deleted)
        self.assertEqual(False, ext_reg.has_ext('A', 'test'))

        b_ext.delete()
        self.assertEqual(True, b_ext.deleted)
        self.assertEqual(False, ext_reg.has_ext('B', 'test'))

    def test_add_ext_multiple(self):
        ext_reg = ExtensionRegistry()
        obj1 = A()
        obj2 = A()
        obj3 = A()
        obj4 = B()
        obj5 = B()
        obj6 = B()

        def obj3_func():
            return obj3

        def obj4_func():
            return obj4

        ext_reg.add_ext('A', 'a1', obj1)
        ext_reg.add_ext('A', 'a2', obj2)
        ext_reg.add_ext_factory('A', 'a3', obj3_func)
        ext_reg.add_ext_factory('B', 'b1', obj4_func)
        ext_reg.add_ext('B', 'b2', obj5)
        ext_reg.add_ext('B', 'b3', obj6)
        self.assertIs(obj1, ext_reg.get_ext_obj('A', 'a1'))
        self.assertIs(obj2, ext_reg.get_ext_obj('A', 'a2'))
        self.assertIs(obj3, ext_reg.get_ext_obj('A', 'a3'))
        self.assertIs(obj4, ext_reg.get_ext_obj('B', 'b1'))
        self.assertIs(obj5, ext_reg.get_ext_obj('B', 'b2'))
        self.assertIs(obj6, ext_reg.get_ext_obj('B', 'b3'))

    def test_find_ext_by_metadata(self):
        ext_reg = ExtensionRegistry()
        obj1 = A()
        obj2 = A()
        obj3 = A()
        obj4 = B()
        obj5 = B()
        obj6 = B()

        class Obj3Factory:
            # noinspection PyMethodMayBeStatic
            def load(self):
                return obj3

        def get_obj4(self):
            return obj4

        ext_reg.add_ext('A', 'a1', obj1, description='knorg')
        ext_reg.add_ext('A', 'a2', obj2, description='gnatz')
        ext_reg.add_ext('A', 'a3', Obj3Factory(), description='gnatz')
        ext_reg.add_ext_factory('B', 'b1', get_obj4, description='knorg')
        ext_reg.add_ext('B', 'b2', obj5, description='gnatz')
        ext_reg.add_ext('B', 'b3', obj6, description='knorg')

        def is_knorg(ext):
            return ext.metadata.get('description') == 'knorg'

        def is_gnatz(ext):
            return ext.metadata.get('description') == 'gnatz'

        ext_list = ext_reg.find_ext('A', predicate=is_knorg)
        self.assertEqual(1, len(ext_list))

        ext_list = ext_reg.find_ext('B', predicate=is_knorg)
        self.assertEqual(2, len(ext_list))

        ext_list = ext_reg.find_ext('A', predicate=is_gnatz)
        self.assertEqual(2, len(ext_list))

        ext_list = ext_reg.find_ext('B', predicate=is_gnatz)
        self.assertEqual(1, len(ext_list))
