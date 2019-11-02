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

import click


# noinspection PyShadowingBuiltins
@click.command(name="dump")
@click.argument('input')
@click.option('--variable', '--var', metavar='VARIABLE', multiple=True,
              help="Name of a variable (multiple allowed).")
@click.option('--encoding', '-E', is_flag=True, flag_value=True,
              help="Dump also variable encoding information.")
def dump(input, variable, encoding):
    """
    Dump contents of an input dataset.
    """
    from xcube.core.dsio import open_dataset
    from xcube.core.dump import dump_dataset
    with open_dataset(input) as ds:
        text = dump_dataset(ds, var_names=variable, show_var_encoding=encoding)
        print(text)
