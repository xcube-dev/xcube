import unittest

from xcube.util.ext import ExtensionRegistry
from xcube.util.plugin import get_plugins, load_plugins, discover_plugin_modules


def init_plugin(ext_registry: ExtensionRegistry):
    """A test plugin that registers test extensions"""
    ext_registry.add_ext(object(), 'test', 'ext1')
    ext_registry.add_ext(object(), 'test', 'ext2')
    ext_registry.add_ext(object(), 'test', 'ext3')


def init_plugin_bad():
    raise RuntimeError()


class EntryPoint:
    def __init__(self, name, func):
        self.name = name
        self.func = func

    def load(self):
        return self.func


class EntryPointBad(EntryPoint):

    def load(self):
        raise RuntimeError()


class PluginTest(unittest.TestCase):

    def setUp(self):
        self.ext_registry = ExtensionRegistry()

    def test_get_xcube_default_plugins(self):
        plugins = get_plugins()
        self.assertIsNotNone(plugins)
        self.assertIn('xcube_default_iproc', plugins)
        self.assertIn('xcube_default_dsio', plugins)
        self.assertIn('xcube_default_cli', plugins)

    def test_load_plugins_by_entry_points(self):
        plugins = load_plugins([EntryPoint('test', init_plugin)], ext_registry=self.ext_registry)
        self.assertEqual(dict(test=dict(name='test', doc='A test plugin that registers test extensions')), plugins)
        self.assertTrue(self.ext_registry.has_ext('test', 'ext1'))
        self.assertTrue(self.ext_registry.has_ext('test', 'ext2'))
        self.assertTrue(self.ext_registry.has_ext('test', 'ext3'))

    def test_load_plugins_by_module_discovery(self):
        entry_points = discover_plugin_modules(module_prefixes=['test'])
        plugins = load_plugins(entry_points, ext_registry=self.ext_registry)
        self.assertEqual(dict(test=dict(name='test', doc='A test plugin that registers test extensions')), plugins)
        self.assertTrue(self.ext_registry.has_ext('test', 'ext1'))
        self.assertTrue(self.ext_registry.has_ext('test', 'ext2'))
        self.assertTrue(self.ext_registry.has_ext('test', 'ext3'))

    def test_load_plugins_by_bad_entry_point(self):
        plugins = load_plugins([EntryPointBad('test', init_plugin)], ext_registry=self.ext_registry)
        self.assertEqual({}, plugins)
        self.assertEqual([], self.ext_registry.get_all_ext_obj('test'))

    def test_load_plugins_by_bad_init_plugin(self):
        plugins = load_plugins([EntryPoint('test', init_plugin_bad)], ext_registry=self.ext_registry)
        self.assertEqual({}, plugins)
        self.assertEqual([], self.ext_registry.get_all_ext_obj('test'))

    def test_load_plugins_init_plugin_not_callable(self):
        plugins = load_plugins([EntryPoint('test', "init_plugin_not_callable")], ext_registry=self.ext_registry)
        self.assertEqual({}, plugins)
        self.assertEqual([], self.ext_registry.get_all_ext_obj('test'))

    def test_load_plugins_by_failing_module_discovery(self):
        entry_points = discover_plugin_modules(module_prefixes=['random'])
        plugins = load_plugins(entry_points, ext_registry=self.ext_registry)
        self.assertEqual({}, plugins)
        self.assertEqual([], self.ext_registry.get_all_ext_obj('test'))
