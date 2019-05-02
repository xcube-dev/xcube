from typing import Any, Dict

import click

from xcube.cli.gen import gen
from xcube.cli.serve import serve
from xcube.cli.grid import grid
from xcube.version import version

DEFAULT_TILE_SIZE = 512


def _parse_kwargs(value: str, metavar: str = None) -> Dict[str, Any]:
    if value:
        try:
            return eval(f"dict({value})", {}, {})
        except Exception:
            if metavar:
                message = f"Invalid value for {metavar}: {value!r}"
            else:
                message = f"Invalid value: {value!r}"
            raise click.ClickException(message)
    else:
        return dict()


# noinspection PyShadowingBuiltins
@click.command(name="extract")
@click.argument('cube', metavar='<cube>')
@click.argument('coords', metavar='<coords>')
@click.option('--indexes', '-i', is_flag=True,
              help="Include indexes in output.")
@click.option('--output', '-o', metavar='<output>',
              help="Output file.")
# @click.option('--format', '-f', metavar='<format>', type=click.Choice(['csv', 'stdout']),
#               help="Format of the output. If not given, guessed from <output>, otherwise <stdout> is used.")
# @click.option('--params', '-p', metavar='<params>',
#               help="Parameters specific for the output format."
#                    " Comma-separated list of <key>=<value> pairs.")
def extract(cube, coords, indexes=False, output=None,
            # format=None, params=None
            ):
    """
    Extract cube time series.
    Extracts data from <cube> at points given by coordinates <coords> and writes the resulting
    time series to <output>.
    """
    import pandas as pd

    cube_path = cube
    coords_path = coords
    output_path = output
    include_indexes = indexes

    from xcube.api import open_dataset, get_cube_values_for_points

    coords = pd.read_csv(coords_path, parse_dates=["time"], infer_datetime_format=True)
    print(coords, [coords[c].values.dtype for c in coords])
    with open_dataset(cube_path) as cube:
        values = get_cube_values_for_points(cube, coords, include_indexes=include_indexes)
        if output_path:
            values.to_csv(output_path)
        else:
            print(values)


# noinspection PyShadowingBuiltins
@click.command(name="chunk")
@click.argument('input', metavar='<input>')
@click.argument('output', metavar='<output>')
@click.option('--format', '-f', metavar='<format>', type=click.Choice(['zarr', 'netcdf']),
              help="Format of the output. If not given, guessed from <output>.")
@click.option('--params', '-p', metavar='<params>',
              help="Parameters specific for the output format."
                   " Comma-separated list of <key>=<value> pairs.")
@click.option('--chunks', '-c', metavar='<chunks>', nargs=1, default=None,
              help='Chunk sizes for each dimension.'
                   ' Comma-separated list of <dim>=<size> pairs,'
                   ' e.g. "time=1,lat=270,lon=270"')
def chunk(input, output, format=None, params=None, chunks=None):
    """
    (Re-)chunk dataset.
    Changes the external chunking of all variables of <input> according to <chunks> and writes
    the result to <output>.
    """
    chunk_sizes = None
    if chunks:
        chunk_sizes = _parse_kwargs(chunks, metavar="<chunks>")
        for k, v in chunk_sizes.items():
            if not isinstance(v, int) or v <= 0:
                raise click.ClickException("Invalid value for <chunks>, "
                                           f"chunk sizes must be positive integers: {chunks}")

    write_kwargs = dict()
    if params:
        write_kwargs = _parse_kwargs(params, metavar="<params>")

    from xcube.util.dsio import guess_dataset_format
    format_name = format if format else guess_dataset_format(output)

    from xcube.api import open_dataset, chunk_dataset, write_dataset

    with open_dataset(input_path=input) as ds:
        if chunk_sizes:
            for k in chunk_sizes:
                if k not in ds.dims:
                    raise click.ClickException("Invalid value for <chunks>, "
                                               f"{k!r} is not the name of any dimension: {chunks}")

        chunked_dataset = chunk_dataset(ds, chunk_sizes=chunk_sizes, format_name=format_name)
        write_dataset(chunked_dataset, output_path=output, format_name=format_name, **write_kwargs)


@click.command(name="level")
@click.argument('input', metavar='<input>')
@click.option('--output', '-o', metavar='<output>',
              help='Output directory. If omitted, "<input>.levels" will be used.')
@click.option('--link', '-l', is_flag=True, flag_value=True,
              help='Link the <input> instead of converting it to a level zero dataset. '
                   'Use with care, as the <input>\'s internal spatial chunk sizes may be inappropriate '
                   'for imaging purposes.')
@click.option('--tile-size', '-t', metavar='<tile-size>',
              help=f'Tile size, given as single integer number or as <tile-width>,<tile-height>. '
              f'If omitted, the tile size will be derived from the <input>\'s '
              f'internal spatial chunk sizes. '
              f'If the <input> is not chunked, tile size will be {DEFAULT_TILE_SIZE}.')
@click.option('--num-levels-max', '-n', metavar='<num-levels-max>', type=int,
              help=f'Maximum number of levels to generate. '
              f'If not given, the number of levels will be derived from '
              f'spatial dimension and tile sizes.')
def level(input, output, link, tile_size, num_levels_max):
    """
    Generate multi-resolution levels.
    Transform the given dataset by <input> into the levels of a multi-level pyramid with spatial resolution
    decreasing by a factor of two in both spatial dimensions and write the result to directory <output>.
    """
    import time
    import os
    from xcube.api.levels import write_levels

    input_path = input
    output_path = output
    link_input = link

    if num_levels_max is not None and num_levels_max < 1:
        raise click.ClickException(f"<num-levels-max> must be a positive integer")

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
            raise click.ClickException("<tile-size> must comprise positive integers")
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
@click.argument('dataset', metavar='<path>')
@click.option('--variable', '-v', metavar='<variable>', multiple=True,
              help="Name of a variable (multiple allowed).")
@click.option('--encoding', '-e', is_flag=True, flag_value=True,
              help="Dump also variable encoding information.")
def dump(dataset, variable, encoding):
    """
    Dump contents of a dataset.
    """
    from xcube.api import open_dataset, dump_dataset
    with open_dataset(dataset) as ds:
        text = dump_dataset(ds, var_names=variable, show_var_encoding=encoding)
        print(text)


# noinspection PyShadowingBuiltins
@click.command(name="vars2dim")
@click.argument('cube', metavar='<cube>')
@click.option('--dim_name', '-d', metavar='<dim-name>',
              default='var',
              help='Name of the new dimension into variables. Defaults to "var".')
@click.option('--var_name', '-v', metavar='<var-name>',
              default='data',
              help='Name of the new variable that includes all variables. Defaults to "data".')
@click.option('--output', '-o', metavar='<output>',
              help="Output file.")
@click.option('--format', '-f', metavar='<format>', type=click.Choice(['zarr', 'netcdf']),
              help="Format of the output. If not given, guessed from <output>.")
def vars2dim(cube, var_name, dim_name, output=None, format=None):
    """
    Convert cube variables into new dimension.
    Moves all variables of <cube> into into a single new variable <var-name>
    with a new dimension <dim-name> and writes the results to <output>.
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
        converted_dataset = vars_to_dim(ds, dim_name=dim_name, var_name=var_name)
        write_dataset(converted_dataset, output_path=output, format_name=format_name)


# noinspection PyShadowingBuiltins
@click.group()
@click.version_option(version)
def cli():
    """
    Xcube Toolkit
    """


cli.add_command(chunk)
cli.add_command(dump)
cli.add_command(extract)
cli.add_command(grid)
cli.add_command(vars2dim)
cli.add_command(gen)
cli.add_command(level)
cli.add_command(serve)


def main(args=None):
    cli.main(args=args)


if __name__ == '__main__':
    main()
