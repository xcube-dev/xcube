# The MIT License (MIT)
# Copyright (c) 2018 by the xcube development team and contributors
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

from xcube.util.cliutil import parse_cli_kwargs
from xcube.webapi import __version__, __description__
from xcube.webapi.defaults import DEFAULT_PORT, DEFAULT_NAME, DEFAULT_ADDRESS, DEFAULT_UPDATE_PERIOD, \
    DEFAULT_TILE_CACHE_SIZE, DEFAULT_TILE_COMP_MODE

__author__ = "Norman Fomferra (Brockmann Consult GmbH)"


@click.command(name='serve')
@click.argument('cubes', metavar='CUBE...', nargs=-1)
@click.version_option(__version__)
@click.option('--name', '-n', metavar='NAME', default=DEFAULT_NAME,
              help=f'Service name. Defaults to {DEFAULT_NAME!r}.')
@click.option('--address', '-a', metavar='ADDRESS', default=DEFAULT_ADDRESS,
              help=f'Service address. Defaults to {DEFAULT_ADDRESS!r}.')
@click.option('--port', '-p', metavar='PORT', default=DEFAULT_PORT, type=int,
              help=f'Port number where the service will listen on. Defaults to {DEFAULT_PORT}.')
@click.option('--update', '-u', metavar='PERIOD', type=float,
              default=DEFAULT_UPDATE_PERIOD,
              help='Service will update after given seconds of inactivity. Zero or a negative value will '
                   'disable update checks. '
                   f'Defaults to {DEFAULT_UPDATE_PERIOD!r}.')
@click.option('--styles', '-s', metavar='STYLES', default=None,
              help='Color mapping styles for variables. '
                   'Used only, if one or more CUBE arguments are provided and CONFIG is not given. '
                   'Comma-separated list with elements of the form '
                   '<var>=(<vmin>,<vmax>) or <var>=(<vmin>,<vmax>,"<cmap>")')
@click.option('--config', '-c', metavar='CONFIG', default=None,
              help='Use datasets configuration file CONFIG. '
                   'Cannot be used if CUBES are provided.')
@click.option('--tilecache', metavar='SIZE', default=DEFAULT_TILE_CACHE_SIZE,
              help=f'In-memory tile cache size in bytes. '
                   f'Unit suffixes {"K"!r}, {"M"!r}, {"G"!r} may be used. '
                   f'Defaults to {DEFAULT_TILE_CACHE_SIZE!r}. '
                   f'The special value {"OFF"!r} disables tile caching.')
@click.option('--tilemode', metavar='MODE', default=None, type=int,
              help='Tile computation mode. '
                   'This is an internal option used to switch between different tile computation implementations. '
                   f'Defaults to {DEFAULT_TILE_COMP_MODE!r}.')
@click.option('--verbose', '-v', is_flag=True,
              help="Delegate logging to the console (stderr).")
@click.option('--traceperf', is_flag=True,
              help="Print performance diagnostics (stdout).")
def serve(cubes: List[str],
          name: str,
          address: str,
          port: int,
          update: float,
          styles: str,
          config: str,
          tilecache: str,
          tilemode: int,
          verbose: bool,
          traceperf: bool):
    """
    Serve data cubes via web service.

    Serves data cubes by a RESTful API and a OGC WMTS 1.0 RESTful and KVP interface.
    The RESTful API documentation can be found at https://app.swaggerhub.com/apis/bcdev/xcube-server.
    """

    if config and cubes:
        raise click.ClickException("CONFIG and CUBES cannot be used at the same time.")
    if styles:
        styles = parse_cli_kwargs(styles, "STYLES")
    from xcube.webapi.app import new_application
    from xcube.webapi.service import Service

    print(f'{__description__}, version {__version__}')
    service = Service(new_application(name),
                      name=name,
                      port=port,
                      address=address,
                      cube_paths=cubes,
                      styles=styles,
                      config_file=config,
                      tile_cache_size=tilecache,
                      tile_comp_mode=tilemode,
                      update_period=update,
                      log_to_stderr=verbose,
                      trace_perf=traceperf)
    service.start()
    return 0


def main(args=None):
    serve.main(args=args)


if __name__ == '__main__':
    main()
