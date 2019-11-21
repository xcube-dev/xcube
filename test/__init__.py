from test.util.test_plugin import init_plugin

# xcube.util.plugin:discover_plugin_modules() will search for "init_test" first (because this package is called "test").
# Only if "init_test" is not found it would search for "init_plugin".
init_test = init_plugin
