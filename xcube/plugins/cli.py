# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import importlib

from xcube.util.ext import ExtensionRegistry


def init_plugin(ext_registry: ExtensionRegistry):
    """
    xcube CLI standard extensions
    """

    cli_command_names = [
        'apply',
        'chunk',
        'dump',
        'extract',
        'gen',
        'grid',
        'level',
        'optimize',
        'prune',
        'resample',
        'serve',
        'timeit',
        'vars2dim',
        'verify',
    ]

    class Factory:
        def __init__(self, name: str):
            self.name = name

        def load(self):
            module = importlib.import_module('xcube.cli.' + self.name)
            return getattr(module, self.name)

    for cli_command_name in cli_command_names:
        ext_registry.add_ext(Factory(cli_command_name), 'cli', cli_command_name)
