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
from typing import List

import click
import re

from xcube.webapi.defaults import DEFAULT_PORT, DEFAULT_ADDRESS, DEFAULT_UPDATE_PERIOD, \
    DEFAULT_TILE_CACHE_SIZE, DEFAULT_TILE_COMP_MODE

__author__ = "Norman Fomferra (Brockmann Consult GmbH)"

VIEWER_ENV_VAR = 'XCUBE_VIEWER_PATH'


@click.command(name='serve')
@click.argument('cube', nargs=-1)
@click.option('--address', '-A', metavar='ADDRESS', default=DEFAULT_ADDRESS,
              help=f'Service address. Defaults to {DEFAULT_ADDRESS!r}.')
@click.option('--port', '-P', metavar='PORT', default=DEFAULT_PORT, type=int,
              help=f'Port number where the service will listen on. Defaults to {DEFAULT_PORT}.')
@click.option('--prefix', metavar='PREFIX',
              help='Service URL prefix. May contain template patterns such as "${version}" or "${name}". '
                   'For example "${name}/api/${version}".')
@click.option('--name', metavar='NAME', hidden=True,
              help='Service name. Deprecated, use prefix option instead.')
@click.option('--update', '-u', 'update_period', metavar='PERIOD', type=float,
              default=DEFAULT_UPDATE_PERIOD,
              help='Service will update after given seconds of inactivity. Zero or a negative value will '
                   'disable update checks. '
                   f'Defaults to {DEFAULT_UPDATE_PERIOD!r}.')
@click.option('--styles', '-S', metavar='STYLES', default=None,
              help='Color mapping styles for variables. '
                   'Used only, if one or more CUBE arguments are provided and CONFIG is not given. '
                   'Comma-separated list with elements of the form '
                   '<var>=(<vmin>,<vmax>) or <var>=(<vmin>,<vmax>,"<cmap>"). '
                   'In order to configure an RGB image based on three variables, '
                   'please use rgb=(Red=(<var>=(<vmin>,<vmax>)),Green=(<var>=(<vmin>,<vmax>)),'
                   'Blue=(<var>=(<vmin>,<vmax>))).')
@click.option('--config', '-c', metavar='CONFIG', default=None,
              help='Use datasets configuration file CONFIG. '
                   'Cannot be used if CUBES are provided.')
@click.option('--tilecache', 'tile_cache_size', metavar='SIZE', default=DEFAULT_TILE_CACHE_SIZE,
              help=f'In-memory tile cache size in bytes. '
                   f'Unit suffixes {"K"!r}, {"M"!r}, {"G"!r} may be used. '
                   f'Defaults to {DEFAULT_TILE_CACHE_SIZE!r}. '
                   f'The special value {"OFF"!r} disables tile caching.')
@click.option('--tilemode', 'tile_comp_mode', metavar='MODE', default=None, type=int,
              help='Tile computation mode. '
                   'This is an internal option used to switch between different tile computation implementations. '
                   f'Defaults to {DEFAULT_TILE_COMP_MODE!r}.')
@click.option('--show', '-s', is_flag=True,
              help=f"Run viewer app. Requires setting the environment variable {VIEWER_ENV_VAR} "
                   f"to a valid xcube-viewer deployment or build directory. "
                   f"Refer to https://github.com/dcs4cop/xcube-viewer for more information.")
@click.option('--verbose', '-v', is_flag=True,
              help="Delegate logging to the console (stderr).")
@click.option('--traceperf', 'trace_perf', is_flag=True,
              help="Print performance diagnostics (stdout).")
@click.option('--aws-prof', 'aws_prof', metavar='PROFILE',
              help="To publish remote CUBEs, use AWS credentials from section "
                   "[PROFILE] found in ~/.aws/credentials.")
@click.option('--aws-env', 'aws_env', is_flag=True,
              help="To publish remote CUBEs, use AWS credentials from environment "
                   "variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
def serve(cube: List[str],
          address: str,
          port: int,
          prefix: str,
          name: str,
          update_period: float,
          styles: str,
          config: str,
          tile_cache_size: str,
          tile_comp_mode: int,
          show: bool,
          verbose: bool,
          trace_perf: bool,
          aws_prof: str,
          aws_env: bool):
    """
    Serve data cubes via web service.

    Serves data cubes by a RESTful API and a OGC WMTS 1.0 RESTful and KVP interface.
    The RESTful API documentation can be found at https://app.swaggerhub.com/apis/bcdev/xcube-server.
    """

    from xcube.cli.common import parse_cli_kwargs
    import os.path

    prefix = prefix or name

    if config and cube:
        raise click.ClickException("CONFIG and CUBES cannot be used at the same time.")
    if styles:
        if 'rgb' in styles:
            styles = _handle_rgb_styles(styles)
        else:
            styles = parse_cli_kwargs(styles, "STYLES")
    if (aws_prof or aws_env) and not cube:
        raise click.ClickException("AWS credentials are only valid in combination with given CUBE argument(s).")

    from xcube.version import version
    from xcube.webapi.defaults import SERVER_NAME, SERVER_DESCRIPTION
    print(f'{SERVER_NAME}: {SERVER_DESCRIPTION}, version {version}')

    if show:
        _run_viewer()

    from xcube.webapi.app import new_application
    from xcube.webapi.service import Service
    service = Service(new_application(prefix, os.path.dirname(config) if config else '.'),
                      prefix=prefix,
                      port=port,
                      address=address,
                      cube_paths=cube,
                      styles=styles,
                      config_file=config,
                      tile_cache_size=tile_cache_size,
                      tile_comp_mode=tile_comp_mode,
                      update_period=update_period,
                      log_to_stderr=verbose,
                      trace_perf=trace_perf,
                      aws_prof=aws_prof,
                      aws_env=aws_env)
    service.start()

    return 0


def _handle_rgb_styles(styles):
    from xcube.cli.common import parse_cli_kwargs

    rgb = re.search(r"rgb=\(Red=\(.*\),Green=\(.*\),Blue=\(.*\)\)", styles)
    if not rgb:
        raise click.ClickException("For a default RGB schema, Red, Green and Blue need to be specified: "
                                   "rgb=(Red=(<var>=(<vmin>,<vmax>)),Green=(<var>=(<vmin>,<vmax>)),"
                                   "Blue=(<var>=(<vmin>,<vmax>)))")

    colors = re.split('rgb=\(', rgb.group())[1][:-1]
    rgb_dict = {'rgb': {}}
    for element in re.findall('[^\)\)]+\)\)', colors):
        for color in ['Red', 'Green', 'Blue']:
            if color in element:
                var_and_range = re.split(f'{color}=\(', element)[1][:-1]
                rgb_dict['rgb'][color] = parse_cli_kwargs(var_and_range, "STYLES")

    none_rgb_vars = styles.split(rgb.group())
    none_rgb_vars = list(filter(any, none_rgb_vars))
    if none_rgb_vars:
        vars_styles = []
        for element in none_rgb_vars:
            var_style = re.search(r".*=\(.*\)", element)
            if var_style.group()[0] == ',':
                vars_styles.append(var_style.group()[1:])
            else:
                vars_styles.append(var_style.group())
        vars_styles = ','.join(vars_styles)
        vars_dict = parse_cli_kwargs(vars_styles, "STYLES")
        rgb_dict.update(vars_dict)
    styles = rgb_dict.copy()
    return styles


def _run_viewer():
    import subprocess
    import threading
    import webbrowser
    import os

    viewer_dir = os.environ.get(VIEWER_ENV_VAR)

    if viewer_dir is None:
        raise click.UsageError('Option "--show": '
                               f"In order to run the viewer, "
                               f"set environment variable {VIEWER_ENV_VAR} "
                               f"to a valid xcube-viewer deployment or build directory.")

    if not os.path.isdir(viewer_dir):
        raise click.UsageError('Option "--show": '
                               f"Viewer path set by environment variable {VIEWER_ENV_VAR} "
                               f"must be a directory: " + viewer_dir)

    def _run():
        print("starting web server...")
        with subprocess.Popen(['python', '-m', 'http.server', '--directory', viewer_dir],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE):
            print("opening viewer...")
            webbrowser.open("http://localhost:8000/index.html")

    threading.Thread(target=_run, name="xcube-viewer-runner").start()


def main(args=None):
    serve.main(args=args)


if __name__ == '__main__':
    main()
