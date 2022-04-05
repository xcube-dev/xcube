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

import sys

import click
from xcube.constants import LOG

# noinspection PyShadowingBuiltins
@click.command(name='verify')
@click.argument('cube')
def verify(cube):
    """
    Perform cube verification.

    \b
    The tool verifies that CUBE
    * defines the dimensions "time", "lat", "lon";
    * has corresponding "time", "lat", "lon" coordinate variables and that they
      are valid, e.g. 1-D, non-empty, using correct units;
    * has valid  bounds variables for "time", "lat", "lon" coordinate
      variables, if any;
    * has any data variables and that they are valid, e.g. min. 3-D, all have
      same dimensions, have at least dimensions "time", "lat", "lon".
    * spatial coordinates and their corresponding bounds (if exist) are equidistant
       and monotonically increasing or decreasing.

    If INPUT is a valid xcube dataset, the tool returns exit code 0.
    Otherwise a violation report is written to stdout and the tool returns exit code 3.
    """
    return _verify(input_path=cube, monitor=print)


def _verify(input_path: str = None, monitor=None):
    from xcube.core.dsio import open_dataset
    from xcube.core.verify import verify_cube

    LOG.info(f'Opening cube from {input_path!r}...')
    with open_dataset(input_path) as cube:
        report = verify_cube(cube)

    if not report:
        monitor("INPUT is a valid cube.")
        return

    monitor('INPUT is not a valid cube due to the following reasons:')
    monitor('- ' + '\n- '.join(report))
    # According to http://tldp.org/LDP/abs/html/exitcodes.html, exit code 3 is not reserved
    sys.exit(3)
