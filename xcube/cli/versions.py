# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
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

from typing import Optional

import os.path

import click

YAML_FORMAT = "yaml"
JSON_FORMAT = "json"
DEFAULT_FORMAT = YAML_FORMAT


# noinspection PyShadowingBuiltins
@click.command(name="versions")
@click.option('--format', '-f', 'format_name',
              metavar='FORMAT',
              type=click.Choice([YAML_FORMAT, JSON_FORMAT],
                                case_sensitive=False),
              help=f'Output format. '
                   f'Must be one of {YAML_FORMAT!r} or {JSON_FORMAT!r}. '
                   f'If not given, derived from OUTPUT name extension.')
@click.option('--output', '-o', 'output_path',
              metavar='OUTPUT',
              help=f'Output file path.')
def versions(format_name: Optional[str], output_path: Optional[str]):
    """
    Get versions of important packages used by xcube.
    """
    from xcube.util.versions import get_xcube_versions
    xcube_versions = get_xcube_versions()

    if not format_name:
        format_name = YAML_FORMAT
        if output_path:
            _, ext = os.path.splitext(output_path)
            if ext.lower() == '.json':
                format_name = JSON_FORMAT

    if format_name.lower() == JSON_FORMAT:
        import json
        text = json.dumps(xcube_versions, indent=2)
    else:
        import yaml
        text = yaml.dump(xcube_versions)

    if output_path:
        with open(output_path, 'w') as fp:
            fp.write(text)
    else:
        print(text)
