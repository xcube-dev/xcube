from typing import Any, Dict

import click

from .version import version


def _parse_kwargs(value: str, metavar: str=None) -> Dict[str, Any]:
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
    Write a new dataset with identical data and compression but with new chunk sizes.
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

    from .api import guess_dataset_format
    format_name = format if format else guess_dataset_format(output)

    from .api import open_dataset, chunk_dataset, write_dataset

    with open_dataset(input_path=input) as ds:
        if chunk_sizes:
            for k in chunk_sizes:
                if k not in ds.dims:
                    raise click.ClickException("Invalid value for <chunks>, "
                                               f"{k!r} is not the name of any dimension: {chunks}")

        chunked_dataset = chunk_dataset(ds, chunk_sizes=chunk_sizes, format_name=format_name)
        write_dataset(chunked_dataset, output_path=output, format_name=format_name, **write_kwargs)


@click.command(name="dump")
@click.argument('dataset', metavar='<path>')
@click.option('--variable', '-v', metavar='<variable>', multiple=True,
              help="Name of a variable (multiple allowed).")
@click.option('--encoding', '-e', is_flag=True, flag_value=True,
              help="Dump also variable encoding information.")
def dump(dataset, variable, encoding):
    """
    Dump contents of dataset.
    """
    from .api import open_dataset, dump_dataset
    with open_dataset(dataset) as ds:
        text = dump_dataset(ds, variable_names=variable, show_var_encoding=encoding)
        print(text)


# noinspection PyShadowingBuiltins
@click.group()
@click.version_option(version)
def cli():
    """
    Xcube Toolkit
    """


cli.add_command(chunk)
cli.add_command(dump)


def main(args=None):
    cli.main(args=args)


if __name__ == '__main__':
    main()
