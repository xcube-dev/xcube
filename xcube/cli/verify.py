# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import sys

import click

from xcube.constants import LOG


# noinspection PyShadowingBuiltins
@click.command(name="verify")
@click.argument("cube")
def verify(cube):
    """Perform cube verification.

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

    LOG.info(f"Opening cube from {input_path!r}...")
    with open_dataset(input_path) as cube:
        report = verify_cube(cube)

    if not report:
        monitor("INPUT is a valid cube.")
        return

    monitor("INPUT is not a valid cube due to the following reasons:")
    monitor("- " + "\n- ".join(report))
    # According to http://tldp.org/LDP/abs/html/exitcodes.html, exit code 3 is not reserved
    sys.exit(3)
