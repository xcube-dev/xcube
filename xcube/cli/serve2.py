# The MIT License (MIT)
# Copyright (c) 2022 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
from typing import List, Optional

import click

from xcube.cli.common import (cli_option_quiet,
                              cli_option_verbosity)

DEFAULT_FRAMEWORK = "tornado"


@click.command(name='serve2')
@click.option('--framework', 'framework_name',
              metavar='FRAMEWORK', default=DEFAULT_FRAMEWORK,
              type=click.Choice(["tornado", "flask"]),
              help=f'Web server framework.'
                   f' Defaults to "{DEFAULT_FRAMEWORK}"')
@click.option('--config', '-c', 'config_paths',
              metavar='CONFIG', default=None, multiple=True,
              help='Configuration file.')
@click.option('--stop-after', 'stop_after',
              metavar='TIME', type=float, default=None,
              help='Stop server after TIME seconds.')
@cli_option_quiet
@cli_option_verbosity
def serve2(framework_name: str,
           config_paths: List[str],
           stop_after: Optional[float],
           quiet: bool,
           verbosity: int):
    """
    Run xcube restful server.
    """
    from xcube.util.config import load_configs
    from xcube.cli.common import configure_cli_output
    from xcube.server.server import Server
    from xcube.server.framework import get_framework_class

    configure_cli_output(quiet=quiet, verbosity=verbosity)

    config = load_configs(*config_paths) if config_paths else {}
    framework = get_framework_class(framework_name)()
    server = Server(framework, config)
    if stop_after is not None:
        server.call_later(stop_after, server.stop)
    server.start()
