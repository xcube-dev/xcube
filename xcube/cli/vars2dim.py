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


# noinspection PyShadowingBuiltins,PyUnusedLocal
@click.command(name="vars2dim")
@click.argument('cube')
@click.option('--variable', '--var', metavar='VARIABLE',
              default='data',
              help='Name of the new variable that includes all variables. Defaults to "data".')
@click.option('--dim_name', '-D', metavar='DIM-NAME',
              default='var',
              help='Name of the new dimension into variables. Defaults to "var".')
@click.option('--output', '-o', metavar='OUTPUT',
              help="Output path. If omitted, 'INPUT-vars2dim.FORMAT' will be used.")
@click.option('--format', '-f', metavar='FORMAT', type=click.Choice(['zarr', 'netcdf']),
              help="Format of the output. If not given, guessed from OUTPUT.")
def vars2dim(cube, variable, dim_name, output=None, format=None):
    """
    Convert cube variables into new dimension.
    Moves all variables of CUBE into a single new variable <var-name>
    with a new dimension DIM-NAME and writes the results to OUTPUT.
    """

    from xcube.core.dsio import guess_dataset_format
    from xcube.core.dsio import open_dataset, write_dataset
    from xcube.core.vars2dim import vars_to_dim
    import os

    if not output:
        dirname = os.path.dirname(cube)
        basename = os.path.basename(cube)
        basename, ext = os.path.splitext(basename)
        output = os.path.join(dirname, basename + '-vars2dim' + ext)

    format_name = format if format else guess_dataset_format(output)

    with open_dataset(input_path=cube) as ds:
        converted_dataset = vars_to_dim(ds, dim_name=dim_name, var_name=variable)
        write_dataset(converted_dataset, output_path=output, format_name=format_name)
