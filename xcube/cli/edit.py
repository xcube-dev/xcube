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

DEFAULT_OUTPUT_PATH = '{input}-edited.zarr'


# noinspection PyShadowingBuiltins
@click.command(name='edit')
@click.argument('cube')
@click.option('--output', '-o', metavar='OUTPUT',
              help=f'Output path. The placeholder "{input}" will be replaced by the input\'s filename '
                   f'without extension (such as ".zarr"). Defaults to "{DEFAULT_OUTPUT_PATH}".',
              default=DEFAULT_OUTPUT_PATH)
@click.option('--metadata', '-m',
              help='The metadata of the cube is edited. '
                   'The metadata to be changed should be saved in a single yml file.')
@click.option('--in-place', '-I', 'in_place',
              help="Edit the cube in place. Ignores output path.",
              is_flag=True)
def edit(cube,
         output,
         metadata,
         in_place):
    """
    Edit the metadata of an xcube dataset.
    Edits the metadata of a given CUBE.
    The command currently works only for data cubes using ZARR format.
    """
    from xcube.api import edit_metadata
    edit_metadata(cube,
                  output_path=output,
                  metadata_path=metadata,
                  in_place=in_place
                  )
