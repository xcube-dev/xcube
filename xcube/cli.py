import click

from .version import version


# noinspection PyShadowingBuiltins
@click.command(name="chunk")
@click.argument('input', metavar='<input>')
@click.argument('output', metavar='<output>')
@click.option('--format', metavar='<format>', type=click.Choice(['zarr', 'netcdf']),
              help="Output format. If not given, guessed from <output>.")
@click.option('--chunks', '-c', metavar='<chunk-sizes>', nargs=1, default=None,
              help='Mapping from dimension name to chunk sizes, e.g. "time=1,lat=270,lon=270"')
def chunk(input, output, format=None, chunks=None):
    """
    Write a new dataset with identical data and compression but with new chunk sizes.
    """
    chunk_sizes = None
    if chunks:
        try:
            chunk_sizes = eval(f"dict({chunks})")
        except (SyntaxError, NameError, TypeError):
            raise click.ClickException(f"Invalid value for option 'chunks': {chunks}")
        for k, v in chunk_sizes.items():
            if not isinstance(v, int) or v <= 0:
                raise click.ClickException("Invalid value for option 'chunks', "
                                           f"chunk sizes must be positive integers: {chunks}")

    from .api import guess_dataset_format
    format_name = format if format else guess_dataset_format(output)

    from .api import open_dataset, chunk_dataset, write_dataset

    with open_dataset(input_path=input) as ds:
        if chunk_sizes:
            for k in chunk_sizes:
                if k not in ds.dims:
                    raise click.ClickException("Invalid value for option 'chunks', "
                                               f"{k!r} is not the name of any dimension: {chunks}")

        chunked_dataset = chunk_dataset(ds, chunk_sizes=chunk_sizes, format_name=format_name)
        write_dataset(chunked_dataset, output_path=output, format_name=format_name)


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
        dump_dataset(ds, variable_names=variable, show_var_encoding=encoding)


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
