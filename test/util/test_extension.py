import unittest
from typing import Any

from xcube.util.extension import ExtensionRegistry, Extension, get_extension_registry
from xcube.util.extension import import_component


class A:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class B(A):
    pass


class ExtensionRegistryTest(unittest.TestCase):
    def test_get_extension_reg(self):
        self.assertIsInstance(get_extension_registry(), ExtensionRegistry)

    def test_get_extension_illegal(self):
        ext_reg = ExtensionRegistry()
        self.assertEqual(False, ext_reg.has_extension('A', 'test'))
        self.assertEqual(None, ext_reg.get_extension('A', 'test'))

    def test_get_component_illegal(self):
        ext_reg = ExtensionRegistry()
        with self.assertRaises(ValueError) as cm:
            ext_reg.get_component('A', 'test')
        self.assertEqual("extension 'test' not found for extension point 'A'", f'{cm.exception}')

    def test_protocol(self):
        ext_reg = ExtensionRegistry()

        a_obj = A()

        def b_loader(ext):
            return B(name=ext.name)

        a_ext = ext_reg.add_extension(component=a_obj, point='A', name='test')
        b_ext = ext_reg.add_extension(loader=b_loader, point='B', name='test')

        self.assertEqual(True, ext_reg.has_extension('A', 'test'))
        self.assertIsInstance(a_ext, Extension)
        self.assertEqual('test', a_ext.name)
        self.assertEqual('A', a_ext.point)
        self.assertEqual(False, a_ext.is_lazy)
        self.assertIs(a_obj, a_ext.component)
        self.assertIs(a_obj, a_ext.component)
        self.assertIs(a_obj, ext_reg.get_component('A', 'test'))
        self.assertIs(a_ext, ext_reg.get_extension('A', 'test'))

        self.assertEqual(True, ext_reg.has_extension('B', 'test'))
        self.assertIsInstance(b_ext, Extension)
        self.assertEqual('test', b_ext.name)
        self.assertEqual('B', b_ext.point)
        self.assertEqual(True, b_ext.is_lazy)
        b_obj = b_ext.component
        self.assertIs(b_obj, b_ext.component)
        self.assertIs(b_obj, ext_reg.get_component('B', 'test'))
        self.assertIs(b_ext, ext_reg.get_extension('B', 'test'))
        self.assertEqual({'name': 'test'}, b_obj.kwargs)

        self.assertEqual([a_ext], ext_reg.find_extensions('A'))
        self.assertEqual([b_ext], ext_reg.find_extensions('B'))
        self.assertEqual([], ext_reg.find_extensions('C'))

        self.assertEqual([a_obj], ext_reg.find_components('A'))
        self.assertEqual([b_obj], ext_reg.find_components('B'))
        self.assertEqual([], ext_reg.find_components('C'))

        ext_reg.remove_extension('A', 'test')
        self.assertEqual(False, ext_reg.has_extension('A', 'test'))

        ext_reg.remove_extension('B', 'test')
        self.assertEqual(False, ext_reg.has_extension('B', 'test'))

    def test_find(self):
        ext_reg = ExtensionRegistry()
        obj1 = A()
        obj2 = A()
        obj3 = A()
        obj4 = B()
        obj5 = B()
        obj6 = B()

        # noinspection PyUnusedLocal
        def load_obj3(ext: Extension):
            return obj3

        # noinspection PyUnusedLocal
        def load_obj4(ext: Extension):
            return obj4

        ext_reg.add_extension(component=obj1, point='A', name='a1', description='knorg')
        ext_reg.add_extension(component=obj2, point='A', name='a2', description='gnatz')
        ext_reg.add_extension(loader=load_obj3, point='A', name='a3', description='gnatz')
        ext_reg.add_extension(loader=load_obj4, point='B', name='b1', description='knorg')
        ext_reg.add_extension(component=obj5, point='B', name='b2', description='gnatz')
        ext_reg.add_extension(component=obj6, point='B', name='b3', description='knorg')

        def is_knorg(ext: Extension):
            return ext.metadata.get('description') == 'knorg'

        def is_gnatz(ext: Extension):
            return ext.metadata.get('description') == 'gnatz'

        result = ext_reg.find_extensions('A', predicate=is_knorg)
        self.assertEqual(1, len(result))
        result = ext_reg.find_components('A', predicate=is_knorg)
        self.assertEqual(1, len(result))
        result = ext_reg.find_extensions('B', predicate=is_knorg)
        self.assertEqual(2, len(result))
        result = ext_reg.find_extensions('C', predicate=is_knorg)
        self.assertEqual(0, len(result))
        result = ext_reg.find_components('C', predicate=is_knorg)
        self.assertEqual(0, len(result))
        result = ext_reg.find_extensions('A', predicate=is_gnatz)
        self.assertEqual(2, len(result))
        result = ext_reg.find_components('A', predicate=is_gnatz)
        self.assertEqual(2, len(result))
        result = ext_reg.find_extensions('B', predicate=is_gnatz)
        self.assertEqual(1, len(result))
        result = ext_reg.find_components('B', predicate=is_gnatz)
        self.assertEqual(1, len(result))
        result = ext_reg.find_extensions('C', predicate=is_gnatz)
        self.assertEqual(0, len(result))
        result = ext_reg.find_components('C', predicate=is_gnatz)
        self.assertEqual(0, len(result))


class TestComponent:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class ImportTest(unittest.TestCase):
    def test_import_component(self):
        loader = import_component('test.util.test_extension:TestComponent')
        self.assertTrue(callable(loader))
        extension = Extension('test_point', 'test_component', component='dummy')
        component = loader(extension)
        self.assertIs(TestComponent, component)

    def test_import_and_transform_component(self):

        def transform(imported_component: Any, loaded_extension_: Extension):
            return imported_component(-1, name=loaded_extension_.name)

        loader = import_component('test.util.test_extension:TestComponent',
                                  transform=transform)
        self.assertTrue(callable(loader))
        extension = Extension('test_point', 'test_component', component='dummy')
        component = loader(extension)
        self.assertIsInstance(component, TestComponent)
        self.assertEqual((-1,), component.args)
        self.assertEqual({'name': 'test_component'}, component.kwargs)

    def test_import_component_and_call(self):
        loader = import_component('test.util.test_extension:TestComponent', call=True, call_args=[42],
                                  call_kwargs={'help': '!'})
        self.assertTrue(callable(loader))
        extension = Extension('test', 'test', component='x')
        component = loader(extension)
        self.assertIsInstance(component, TestComponent)
        self.assertEqual((42,), component.args)
        self.assertEqual({'help': '!'}, component.kwargs)
