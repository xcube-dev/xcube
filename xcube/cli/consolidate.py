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

DEFAULT_OUTPUT_PATH = '{input}-consolidated.zarr'


# noinspection PyShadowingBuiltins
@click.command(name='cons')
@click.argument('input')
@click.option('--output', '-o', 'output_path',
              help=f'Output path. Defaults to "{DEFAULT_OUTPUT_PATH}".',
              default=DEFAULT_OUTPUT_PATH)
@click.option('--in-place', '-I', 'in_place',
              help="Consolidate cube in place. Ignores output path.",
              is_flag=True)
@click.option('--unchunk', '-U', 'unchunk_coords',
              help="Convert any chunked arrays of coordinate variables into a single, non-chunked, contiguous arrays.",
              is_flag=True)
def consolidate(input,
                output_path,
                in_place,
                unchunk_coords):
    """
    Consolidate data cube files.

    Reduces the number of metadata and coordinate data files in data cube given by INPUT.
    Consolidated cubes open much faster from remote locations, e.g. in object storage,
    because obviously much less HTTP requests are required to fetch initial cube meta
    information.

    That is, it merges all metadata files into a single top-level JSON file ".zmetadata".
    Optionally, it removes any chunking of coordinate variables
    so they comprise a single binary data file instead of one file per data chunk.

    The primary usage of this tool is to reduce the number of metadata and coordinate files
    so opening cubes from remote locations, e.g. in object storage, requires much less HTTP
    requests.
    """
    from xcube.api import consolidate_dataset
    consolidate_dataset(input,
                        output_path=output_path,
                        in_place=in_place,
                        unchunk_coords=unchunk_coords)
