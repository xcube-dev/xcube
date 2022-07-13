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

import json
from typing import List, Optional, Dict, Tuple, Callable

import click
import yaml

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
@click.option('--base-dir', 'base_dir',
              metavar='BASE_DIR', default=None,
              help='Directory used to resolve relative paths'
                   ' in CONFIG files. Defaults to the parent directory'
                   ' of (last) CONFIG file.')
@click.option('--prefix',
              metavar='PREFIX', default=None,
              help='Prefix to be used for all endpoint URLs.')
@click.option('--traceperf', is_flag=True, default=None,
              help='Whether to output extra performance logs.')
@click.option('--update-after', 'update_after',
              metavar='TIME', type=float, default=None,
              help='Check for server configuration updates every'
                   ' TIME seconds.')
@click.option('--stop-after', 'stop_after',
              metavar='TIME', type=float, default=None,
              help='Unconditionally stop service after TIME seconds.')
@cli_option_quiet
@cli_option_verbosity
@click.argument('command', metavar='[COMMAND]', nargs=-1)
def serve2(framework_name: str,
           port: int,
           address: str,
           config_paths: List[str],
           base_dir: Optional[str],
           prefix: Optional[str],
           traceperf: Optional[bool],
           update_after: Optional[float],
           stop_after: Optional[float],
           quiet: bool,
           verbosity: int,
           command: List[str]):
    """
    Run the xcube Server.

    The optional COMMAND is one of the following

    \b
    - "list apis" lists the APIs provided by the server
    - "show openapi" outputs the OpenAPI document representing this server
    - "show config" outputs the current server configuration
    - "show configschema" outputs the JSON Schema for the server configuration

    The "show" commands may be suffixed by "yaml" or "json"
    forcing the respective output format. The default format is YAML.

    If COMMAND is provided, the server will not start.
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

    if base_dir is not None:
        config["base_dir"] = base_dir
    elif "base_dir" not in config and config_paths:
        import os.path
        config["base_dir"] = os.path.abspath(os.path.dirname(config_paths[-1]))

    if prefix is not None:
        config["prefix"] = prefix

    if traceperf is not None:
        config["trace_perf"] = traceperf

    framework = get_framework_class(framework_name)()
    server = Server(framework, config)

    if command:
        return exec_command(server, command)

    if update_after is not None:
        change_observer = ConfigChangeObserver(server,
                                               config_paths,
                                               update_after)
        server.call_later(update_after, change_observer.check)

    if stop_after is not None:
        server.call_later(stop_after, server.stop)

    server.start()


def exec_command(server, command):
    def to_yaml(obj):
        yaml.safe_dump(obj, sys.stdout, indent=2)

    def to_json(obj):
        json.dump(obj, sys.stdout, indent=2)

    available_formats = {
        'yaml': to_yaml,
        'json': to_json
    }

    format_name = 'yaml'
    import sys
    if len(command) == 3:
        format_name = command[2]
        command = command[:2]
    if format_name.lower() not in available_formats:
        raise click.ClickException(
            f'Invalid format {format_name}.'
            f' Must be one of {", ".join(available_formats.keys())}.'
        )
    output_fn = available_formats[format_name.lower()]

    def list_apis():
        for api in server.apis:
            print(f'{api.name} - {api.description}')

    def show_open_api():
        output_fn(server.ctx.open_api_doc)

    def show_config():
        output_fn(server.ctx.config)

    def show_config_schema():
        output_fn(server.config_schema.to_dict())

    available_commands: Dict[Tuple[str, ...], Callable[[], None]] = {
        ("list", "apis"): list_apis,
        ("show", "openapi"): show_open_api,
        ("show", "config"): show_config,
        ("show", "configschema"): show_config_schema,
    }

    command_fn = available_commands.get(tuple(c.lower() for c in command))
    if command_fn is None:
        raise click.ClickException(
            f'Invalid command {" ".join(command)}. '
            f'Possible commands are '
            f'{", ".join(" ".join(k) for k in available_commands.keys())}.'
        )
    return command_fn()
