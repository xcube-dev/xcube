# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import click


# noinspection PyShadowingBuiltins
@click.command(name="dump")
@click.argument("input")
@click.option(
    "--variable",
    "--var",
    metavar="VARIABLE",
    multiple=True,
    help="Name of a variable (multiple allowed).",
)
@click.option(
    "--encoding",
    "-E",
    is_flag=True,
    flag_value=True,
    help="Dump also variable encoding information.",
)
def dump(input, variable, encoding):
    """Dump contents of an input dataset."""
    from xcube.core.dsio import open_dataset
    from xcube.core.dump import dump_dataset

    with open_dataset(input) as ds:
        text = dump_dataset(ds, var_names=variable, show_var_encoding=encoding)
        print(text)
