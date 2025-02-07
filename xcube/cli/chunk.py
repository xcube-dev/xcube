# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import click

from xcube.cli.common import (
    cli_option_quiet,
    cli_option_verbosity,
    configure_cli_output,
    parse_cli_kwargs,
)

DEFAULT_OUTPUT_PATH = "out.zarr"


# noinspection PyShadowingBuiltins
@click.command(name="chunk")
@click.argument("cube")
@click.option(
    "--output",
    "-o",
    metavar="OUTPUT",
    default=DEFAULT_OUTPUT_PATH,
    help=f"Output path. Defaults to {DEFAULT_OUTPUT_PATH!r}",
)
@click.option(
    "--format",
    "-f",
    metavar="FORMAT",
    type=click.Choice(["zarr", "netcdf"]),
    help="Format of the output. If not given, guessed from OUTPUT.",
)
@click.option(
    "--params",
    "-p",
    metavar="PARAMS",
    help="Parameters specific for the output format."
    " Comma-separated list of <key>=<value> pairs.",
)
@click.option(
    "--chunks",
    "-C",
    metavar="CHUNKS",
    nargs=1,
    default=None,
    help="Chunk sizes for each dimension."
    " Comma-separated list of <dim>=<size> pairs,"
    ' e.g. "time=1,lat=270,lon=270"',
)
@cli_option_quiet
@cli_option_verbosity
def chunk(
    cube, output, format=None, params=None, chunks=None, quiet=None, verbosity=None
):
    """
    (Re-)chunk xcube dataset.
    Changes the external chunking of all variables of CUBE according to CHUNKS and writes
    the result to OUTPUT.

    Note: There is a possibly more efficient way to (re-)chunk datasets through the
    dedicated tool "rechunker", see https://rechunker.readthedocs.io.
    """
    configure_cli_output(quiet=quiet, verbosity=verbosity)

    chunk_sizes = None
    if chunks:
        chunk_sizes = parse_cli_kwargs(chunks, metavar="CHUNKS")
        for k, v in chunk_sizes.items():
            if not isinstance(v, int) or v <= 0:
                raise click.ClickException(
                    "Invalid value for CHUNKS, "
                    f"chunk sizes must be positive integers: {chunks}"
                )

    write_kwargs = dict()
    if params:
        write_kwargs = parse_cli_kwargs(params, metavar="PARAMS")

    from xcube.core.chunk import chunk_dataset
    from xcube.core.dsio import guess_dataset_format, open_dataset, write_dataset

    format_name = format if format else guess_dataset_format(output)

    with open_dataset(input_path=cube) as ds:
        if chunk_sizes:
            for k in chunk_sizes:
                if k not in ds.sizes:
                    raise click.ClickException(
                        "Invalid value for CHUNKS, "
                        f"{k!r} is not the name of any dimension: {chunks}"
                    )

        chunked_dataset = chunk_dataset(
            ds, chunk_sizes=chunk_sizes, format_name=format_name
        )
        write_dataset(
            chunked_dataset, output_path=output, format_name=format_name, **write_kwargs
        )
