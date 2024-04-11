# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from test.util.test_plugin import init_plugin

# xcube.util.plugin:discover_plugin_modules() will search for "init_test" first (because this package is called "test").
# Only if "init_test" is not found it would search for "init_plugin".
init_test = init_plugin
