# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Optional, Tuple

import click

from xcube.cli.common import (
    cli_option_quiet,
    cli_option_verbosity,
    configure_cli_output,
)
from xcube.constants import LOG

DEFAULT_TILE_SIZE = 512
DEFAULT_AGG_METHOD = "first"


# noinspection PyShadowingBuiltins
@click.command(name="level")
@click.argument("input")
@click.option(
    "--output",
    "-o",
    metavar="OUTPUT",
    help='Output path. If omitted, "INPUT.levels" will be used.'
    " You can also use S3 object storage URLs of the form"
    ' "s3://<bucket>/<path>" or "https://<endpoint>"',
)
@click.option(
    "--link",
    "-L",
    is_flag=True,
    flag_value=True,
    help="Link the INPUT instead of converting it to a level"
    " zero dataset. Use with care, as the INPUT's internal"
    " spatial chunk sizes may be inappropriate"
    " for imaging purposes.",
)
@click.option(
    "--tile-size",
    "-t",
    metavar="TILE_SIZE",
    help=f"Tile size, given as single integer number or"
    f" as <tile-width>,<tile-height>."
    f" If omitted, the tile size will be derived"
    f" from the INPUT's"
    f" internal spatial chunk sizes."
    f" If the INPUT is not chunked,"
    f" tile size will be {DEFAULT_TILE_SIZE}.",
)
@click.option(
    "--num-levels-max",
    "-n",
    metavar="NUM_LEVELS_MAX",
    type=int,
    help=f"Maximum number of levels to generate."
    f" If not given, the number of levels will"
    f" be derived from spatial dimension and tile sizes.",
)
@click.option(
    "--agg-methods",
    "-A",
    metavar="AGG_METHODS",
    default=DEFAULT_AGG_METHOD,
    help=f"Aggregation method(s) to be used for data variables."
    f' Either one of "first", "min", "max", "mean", "median",'
    f' "mode", "auto" or list of assignments to individual'
    f" variables using the notation"
    f' "<var1>=<method1>,<var2>=<method2>,..."'
    f' Defaults to "{DEFAULT_AGG_METHOD}".',
)
@click.option(
    "--replace",
    "-r",
    is_flag=True,
    flag_value=True,
    help=f"Whether to replace an existing dataset at OUTPUT.",
)
@click.option(
    "--anon",
    "-a",
    is_flag=True,
    flag_value=True,
    help=f"For S3 inputs or outputs, whether the access"
    f" is anonymous. By default, credentials are required.",
)
@cli_option_quiet
@cli_option_verbosity
def level(
    input: str,
    output: Optional[str],
    link: bool,
    tile_size: Optional[str],
    num_levels_max: int,
    agg_methods: str,
    replace: bool,
    anon: bool,
    quiet: bool,
    verbosity: int,
):
    """Generate multi-resolution levels.

    Transform the given dataset by INPUT into the levels of a
    multi-level pyramid with spatial resolution decreasing by a
    factor of two in both spatial dimensions and write the
    result to directory OUTPUT.

    INPUT may be an S3 object storage URL of the form
    "s3://<bucket>/<path>" or "https://<endpoint>".
    """
    import os
    import time

    from xcube.cli.common import assert_positive_int_item, parse_cli_sequence
    from xcube.core.store import new_fs_data_store
    from xcube.core.subsampling import assert_valid_agg_methods

    configure_cli_output(quiet=quiet, verbosity=verbosity)

    input_path = input
    output_path = output
    link_input = link

    if tile_size is not None:
        tile_size = parse_cli_sequence(
            tile_size,
            metavar="TILE_SIZE",
            num_items=2,
            item_plural_name="tile sizes",
            item_parser=int,
            item_validator=assert_positive_int_item,
        )

    if num_levels_max is not None and num_levels_max < 1:
        raise click.ClickException(f"NUM_LEVELS_MAX must be a positive integer")

    try:
        if "=" in agg_methods:
            agg_methods = {
                p[0].strip(): (p[1].strip() if len(p) == 2 else None)
                for p in (c.split("=", maxsplit=2) for c in agg_methods.split(","))
            }
        assert_valid_agg_methods(agg_methods)
    except (TypeError, ValueError, SyntaxError) as e:
        raise click.ClickException("invalid AGG_METHODS") from e

    input_protocol, input_path = _split_protocol_and_path(input_path)

    if not output_path:
        dir_path = os.path.dirname(input_path).replace("\\", "/")
        basename, ext = os.path.splitext(os.path.basename(input_path))
        output_protocol = input_protocol
        output_path = f"{dir_path}/{basename + '.levels'}"
    else:
        output_protocol, output_path = _split_protocol_and_path(output_path)

    if link_input and input_protocol != output_protocol:
        raise click.ClickException(
            "Links can be used only if input and output are in the same filesystem"
        )

    start_time = time.time()

    input_store = new_fs_data_store(
        input_protocol,
        storage_options=dict(anon=anon) if input_protocol == "s3" else None,
    )
    input_dataset = input_store.open_data(input_path)

    output_store = new_fs_data_store(
        output_protocol,
        storage_options=dict(anon=anon) if output_protocol == "s3" else None,
    )
    try:
        output_store.write_data(
            input_dataset,
            output_path,
            replace=replace,
            tile_size=tile_size,
            num_levels=num_levels_max,
            base_dataset_id=input_path if link_input else None,
            agg_methods=agg_methods,
        )
    except FileExistsError as e:
        raise click.ClickException(f"output {output_path!r} already exists") from e

    LOG.info(
        f"Multi-level dataset written to {output_path}"
        f" after {time.time() - start_time} seconds"
    )


def _split_protocol_and_path(path) -> tuple[str, str]:
    if "://" in path:
        protocol, path = path.split("://", 2)
        if protocol == "https":
            protocol = "s3"
    else:
        protocol = "file"
    return protocol, path
