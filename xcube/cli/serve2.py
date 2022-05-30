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
                              cli_option_verbosity,
                              configure_cli_output)
from xcube.constants import (DEFAULT_SERVER_FRAMEWORK,
                             DEFAULT_SERVER_PORT,
                             DEFAULT_SERVER_ADDRESS)


@click.command(name='serve2')
@click.option('--framework', 'framework_name',
              metavar='FRAMEWORK', default=DEFAULT_SERVER_FRAMEWORK,
              type=click.Choice(["tornado", "flask"]),
              help=f'Web server framework.'
                   f' Defaults to "{DEFAULT_SERVER_FRAMEWORK}"')
@click.option('--port', '-p',
              metavar='PORT', default=None, type=int,
              help=f'Service port number.'
                   f' Defaults to {DEFAULT_SERVER_PORT}')
@click.option('--address', '-a',
              metavar='ADDRESS', default=None,
              help=f'Service address.'
                   f' Defaults to "{DEFAULT_SERVER_ADDRESS}".')
@click.option('--config', '-c', 'config_paths',
              metavar='CONFIG', default=None, multiple=True,
              help='Configuration YAML or JSON file. '
                   ' If multiple are passed,'
                   ' they will be merged in order.')
@click.option('--update-after', 'update_after',
              metavar='TIME', type=float, default=None,
              help='Check for server configuration updates every'
                   ' TIME seconds.')
@click.option('--stop-after', 'stop_after',
              metavar='TIME', type=float, default=None,
              help='Unconditionally stop service after TIME seconds.')
@cli_option_quiet
@cli_option_verbosity
def serve2(framework_name: str,
           port: int,
           address: str,
           config_paths: List[str],
           update_after: Optional[float],
           stop_after: Optional[float],
           quiet: bool,
           verbosity: int):
    """
    Run xcube restful server.
    """
    from xcube.server.framework import get_framework_class
    from xcube.server.helpers import ConfigChangeObserver
    from xcube.server.server import Server
    from xcube.util.config import load_configs

    configure_cli_output(quiet=quiet, verbosity=verbosity)

    config = load_configs(*config_paths) if config_paths else {}
    if port is not None:
        config["port"] = port
    if address is not None:
        config["address"] = address

    framework = get_framework_class(framework_name)()
    server = Server(framework, config)

    if update_after is not None:
        change_observer = ConfigChangeObserver(server,
                                               config_paths,
                                               update_after)
        server.call_later(update_after, change_observer.check)

    if stop_after is not None:
        server.call_later(stop_after, server.stop)

    server.start()
