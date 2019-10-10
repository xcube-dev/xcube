import sys

import click

from xcube.api.gen.defaults import DEFAULT_OUTPUT_PATH
from xcube.cli.apply import apply
from xcube.cli.edit import edit
from xcube.cli.extract import extract
from xcube.cli.gen import gen
from xcube.cli.grid import grid
from xcube.cli.optimize import optimize
from xcube.cli.prune import prune
from xcube.cli.resample import resample
from xcube.cli.serve import serve
from xcube.cli.timeit import timeit
from xcube.cli.verify import verify
from xcube.util.cliutil import cli_option_scheduler, cli_option_traceback, handle_cli_exception, new_cli_ctx_obj, \
    parse_cli_kwargs
from xcube.version import version


# noinspection PyShadowingBuiltins
@click.command(name="chunk")
@click.argument('cube')
@click.option('--output', '-o', metavar='OUTPUT', default=DEFAULT_OUTPUT_PATH,
              help=f'Output path. Defaults to {DEFAULT_OUTPUT_PATH!r}')
@click.option('--format', '-f', metavar='FORMAT', type=click.Choice(['zarr', 'netcdf']),
              help="Format of the output. If not given, guessed from OUTPUT.")
@click.option('--params', '-p', metavar='PARAMS',
              help="Parameters specific for the output format."
                   " Comma-separated list of <key>=<value> pairs.")
@click.option('--chunks', '-C', metavar='CHUNKS', nargs=1, default=None,
              help='Chunk sizes for each dimension.'
                   ' Comma-separated list of <dim>=<size> pairs,'
                   ' e.g. "time=1,lat=270,lon=270"')
def chunk(cube, output, format=None, params=None, chunks=None):
    """
    (Re-)chunk xcube dataset.
    Changes the external chunking of all variables of CUBE according to CHUNKS and writes
    the result to OUTPUT.
    """
    chunk_sizes = None
    if chunks:
        chunk_sizes = parse_cli_kwargs(chunks, metavar="CHUNKS")
        for k, v in chunk_sizes.items():
            if not isinstance(v, int) or v <= 0:
                raise click.ClickException("Invalid value for CHUNKS, "
                                           f"chunk sizes must be positive integers: {chunks}")

    write_kwargs = dict()
    if params:
        write_kwargs = parse_cli_kwargs(params, metavar="PARAMS")

    from xcube.util.dsio import guess_dataset_format
    format_name = format if format else guess_dataset_format(output)

    from xcube.api import open_dataset, chunk_dataset, write_dataset

    with open_dataset(input_path=cube) as ds:
        if chunk_sizes:
            for k in chunk_sizes:
                if k not in ds.dims:
                    raise click.ClickException("Invalid value for CHUNKS, "
                                               f"{k!r} is not the name of any dimension: {chunks}")

        chunked_dataset = chunk_dataset(ds, chunk_sizes=chunk_sizes, format_name=format_name)
        write_dataset(chunked_dataset, output_path=output, format_name=format_name, **write_kwargs)


DEFAULT_TILE_SIZE = 512


# noinspection PyShadowingBuiltins
@click.command(name="level")
@click.argument('input')
@click.option('--output', '-o', metavar='OUTPUT',
              help='Output path. If omitted, "INPUT.levels" will be used.')
@click.option('--link', '-L', is_flag=True, flag_value=True,
              help='Link the INPUT instead of converting it to a level zero dataset. '
                   'Use with care, as the INPUT\'s internal spatial chunk sizes may be inappropriate '
                   'for imaging purposes.')
@click.option('--tile-size', '-t', metavar='TILE-SIZE',
              help=f'Tile size, given as single integer number or as <tile-width>,<tile-height>. '
                   f'If omitted, the tile size will be derived from the INPUT\'s '
                   f'internal spatial chunk sizes. '
                   f'If the INPUT is not chunked, tile size will be {DEFAULT_TILE_SIZE}.')
@click.option('--num-levels-max', '-n', metavar='NUM-LEVELS-MAX', type=int,
              help=f'Maximum number of levels to generate. '
                   f'If not given, the number of levels will be derived from '
                   f'spatial dimension and tile sizes.')
def level(input, output, link, tile_size, num_levels_max):
    """
    Generate multi-resolution levels.
    Transform the given dataset by INPUT into the levels of a multi-level pyramid with spatial resolution
    decreasing by a factor of two in both spatial dimensions and write the result to directory OUTPUT.
    """
    import time
    import os
    from xcube.api.levels import write_levels

    input_path = input
    output_path = output
    link_input = link

    if num_levels_max is not None and num_levels_max < 1:
        raise click.ClickException(f"NUM-LEVELS-MAX must be a positive integer")

    if not output_path:
        basename, ext = os.path.splitext(input_path)
        output_path = os.path.join(os.path.dirname(input_path), basename + ".levels")

    if os.path.exists(output_path):
        raise click.ClickException(f"output {output_path!r} already exists")

    spatial_tile_shape = None
    if tile_size is not None:
        try:
            tile_size = int(tile_size)
            tile_size = tile_size, tile_size
        except ValueError:
            tile_size = map(int, tile_size.split(","))
            if tile_size != 2:
                raise click.ClickException("Expected a pair of positive integers <tile-width>,<tile-height>")
        if tile_size[0] < 1 or tile_size[1] < 1:
            raise click.ClickException("TILE-SIZE must comprise positive integers")
        spatial_tile_shape = tile_size[1], tile_size[0]

    start_time = t0 = time.perf_counter()

    # noinspection PyUnusedLocal
    def progress_monitor(dataset, index, num_levels):
        nonlocal t0
        print(f"level {index + 1} of {num_levels} written after {time.perf_counter() - t0} seconds")
        t0 = time.perf_counter()

    levels = write_levels(output_path,
                          input_path=input_path,
                          link_input=link_input,
                          progress_monitor=progress_monitor,
                          spatial_tile_shape=spatial_tile_shape,
                          num_levels_max=num_levels_max)
    print(f"{len(levels)} level(s) written into {output_path} after {time.perf_counter() - start_time} seconds")


@click.command(name="dump")
@click.argument('input')
@click.option('--variable', '--var', metavar='VARIABLE', multiple=True,
              help="Name of a variable (multiple allowed).")
@click.option('--encoding', '-E', is_flag=True, flag_value=True,
              help="Dump also variable encoding information.")
def dump(input, variable, encoding):
    """
    Dump contents of an input dataset.
    """
    from xcube.api import open_dataset, dump_dataset
    with open_dataset(input) as ds:
        text = dump_dataset(ds, var_names=variable, show_var_encoding=encoding)
        print(text)


# noinspection PyShadowingBuiltins,PyUnusedLocal
@click.command(name="vars2dim")
@click.argument('cube')
@click.option('--variable', '--var', metavar='VARIABLE',
              default='data',
              help='Name of the new variable that includes all variables. Defaults to "data".')
@click.option('--dim_name', '-D', metavar='DIM-NAME',
              default='var',
              help='Name of the new dimension into variables. Defaults to "var".')
@click.option('--output', '-o', metavar='OUTPUT',
              help="Output path. If omitted, 'INPUT-vars2dim.INPUT-FORMAT' will be used.")
@click.option('--format', '-f', metavar='FORMAT', type=click.Choice(['zarr', 'netcdf']),
              help="Format of the output. If not given, guessed from OUTPUT.")
def vars2dim(cube, variable, dim_name, output=None, format=None):
    """
    Convert cube variables into new dimension.
    Moves all variables of CUBE into into a single new variable <var-name>
    with a new dimension DIM-NAME and writes the results to OUTPUT.
    """

    from xcube.util.dsio import guess_dataset_format
    from xcube.api import open_dataset, vars_to_dim, write_dataset
    import os

    if not output:
        dirname = os.path.dirname(cube)
        basename = os.path.basename(cube)
        basename, ext = os.path.splitext(basename)
        output = os.path.join(dirname, basename + '-vars2dim' + ext)

    format_name = format if format else guess_dataset_format(output)

    with open_dataset(input_path=cube) as ds:
        converted_dataset = vars_to_dim(ds, dim_name=dim_name, var_name=variable)
        write_dataset(converted_dataset, output_path=output, format_name=format_name)


# noinspection PyShadowingBuiltins,PyUnusedLocal
@click.group()
@click.version_option(version)
@cli_option_traceback
@cli_option_scheduler
def cli(traceback=False, scheduler=None):
    """
    Xcube Toolkit
    """


cli.add_command(apply)
cli.add_command(chunk)
cli.add_command(optimize)
cli.add_command(dump)
cli.add_command(edit)
cli.add_command(extract)
cli.add_command(gen)
cli.add_command(grid)
cli.add_command(level)
cli.add_command(prune)
cli.add_command(resample)
cli.add_command(serve)
cli.add_command(timeit)
cli.add_command(vars2dim)
cli.add_command(verify)


def main(args=None):
    # noinspection PyBroadException
    ctx_obj = new_cli_ctx_obj()
    try:
        exit_code = cli.main(args=args, obj=ctx_obj, standalone_mode=False)
    except Exception as e:
        exit_code = handle_cli_exception(e, traceback_mode=ctx_obj.get("traceback", False))
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
