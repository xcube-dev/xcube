# The MIT License (MIT)
# Copyright (c) 2020 by the xcube development team and contributors
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


@click.command(name="gen2")
@click.argument('request_path', type=str, required=False, metavar='REQUEST')
@click.option('--output', '-o', 'output_path', metavar='OUTPUT',
              help='Output ZARR directory in local file system. '
                   'Overwrites output configuration in REQUEST if given.')
@click.option('--callback', '--cb', 'callback_api_url', metavar='URL',
              help='Optional URL of an API for reporting status information. '
                   'The URL must accept the PUT method and support JSON bodies.')
@click.option('--verbose', '-v',
              is_flag=True,
              multiple=True,
              help='Control amount of information dumped to stdout.')
def gen2(request_path: str,
         output_path: str = None,
         callback_api_url=None,
         verbose: bool = False):
    """
    Generate a data cube.

    Creates a cube view from one or more cube stores, optionally performs some cube transformation,
    and writes the resulting cube to some target cube store.

    REQUEST is the cube generation request. It may be provided as a JSON or YAML file
    (file extensions ".json" or ".yaml"). If the REQUEST file argument is omitted, it is expected that
    the Cube generation request is piped as a JSON string.
    """
    # noinspection PyProtectedMember
    from xcube.cli._gen2.main import main
    main(request_path,
         verbose=verbose,
         exception_type=click.ClickException)
