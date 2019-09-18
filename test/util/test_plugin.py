import unittest

from xcube.util.plugin import get_plugins, load_plugins

PLUGIN_INIT = False


def init_plugin_and_succeed():
    global PLUGIN_INIT
    PLUGIN_INIT = True


def init_plugin_but_fail():
    raise RuntimeError()


class EntryPoint:
    def __init__(self, name, func):
        self.name = name
        self.func = func

    def load(self):
        return self.func


class BadEntryPoint(EntryPoint):

    def load(self):
        raise RuntimeError()


class PluginTest(unittest.TestCase):

    def test_get_plugins(self):
        self.assertIsNotNone(get_plugins())

    def test_load_plugins(self):
        global PLUGIN_INIT
        PLUGIN_INIT = False
        plugins = load_plugins([EntryPoint('test', init_plugin_and_succeed)])
        self.assertEqual(dict(test=dict(entry_point='test')), plugins)
        self.assertEqual(True, PLUGIN_INIT)

    def test_load_plugins_fail_load(self):
        plugins = load_plugins([BadEntryPoint('test', init_plugin_and_succeed)])

        self.assertEqual({}, plugins)

    def test_load_plugins_fail_call(self):
        plugins = load_plugins([EntryPoint('test', init_plugin_but_fail)])

        self.assertEqual({}, plugins)

    def test_load_plugins_not_callable(self):
        plugins = load_plugins([EntryPoint('test', "init_plugin_and_succeed")])

        self.assertEqual({}, plugins)
