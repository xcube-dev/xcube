# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import warnings
from collections.abc import Sequence
from typing import Any, Dict

import click

from xcube.cli.common import (
    cli_option_quiet,
    cli_option_verbosity,
    configure_cli_output,
)
from xcube.constants import FORMAT_NAME_MEM, FORMAT_NAME_NETCDF4, FORMAT_NAME_ZARR, LOG

UPSAMPLING_METHODS = ["asfreq", "ffill", "bfill", "pad", "nearest", "interpolate"]
DOWNSAMPLING_METHODS = [
    "count",
    "first",
    "last",
    "min",
    "max",
    "sum",
    "prod",
    "mean",
    "median",
    "std",
    "var",
]
RESAMPLING_METHODS = UPSAMPLING_METHODS + DOWNSAMPLING_METHODS

SPLINE_INTERPOLATION_KINDS = ["zero", "slinear", "quadratic", "cubic"]
OTHER_INTERPOLATION_KINDS = ["linear", "nearest", "previous", "next"]
INTERPOLATION_KINDS = SPLINE_INTERPOLATION_KINDS + OTHER_INTERPOLATION_KINDS

OUTPUT_FORMAT_NAMES = [FORMAT_NAME_ZARR, FORMAT_NAME_NETCDF4, FORMAT_NAME_MEM]

DEFAULT_OUTPUT_PATH = "out.zarr"
DEFAULT_RESAMPLING_METHOD = "mean"
DEFAULT_RESAMPLING_FREQUENCY = "1D"
DEFAULT_RESAMPLING_BASE = None  # Deprecated since 1.0.4!
DEFAULT_INTERPOLATION_KIND = "linear"


# noinspection PyShadowingBuiltins
@click.command(name="resample")
@click.argument("cube")
@click.option(
    "--config",
    "-c",
    metavar="CONFIG",
    multiple=True,
    help="xcube dataset configuration file in YAML format. More than one config input file is allowed."
    "When passing several config files, they are merged considering the order passed via command line.",
)
@click.option(
    "--output",
    "-o",
    metavar="OUTPUT",
    default=DEFAULT_OUTPUT_PATH,
    help=f"Output path. Defaults to {DEFAULT_OUTPUT_PATH!r}.",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(OUTPUT_FORMAT_NAMES),
    help="Output format. If omitted, format will be guessed from output path.",
)
@click.option(
    "--variables",
    "--vars",
    metavar="VARIABLES",
    help="Comma-separated list of names of variables to be included.",
)
@click.option(
    "--method",
    "-M",
    multiple=True,
    help=f"Temporal resampling method. "
    f"Available downsampling methods are "
    f"{', '.join(map(repr, DOWNSAMPLING_METHODS))}, "
    f"the upsampling methods are "
    f"{', '.join(map(repr, UPSAMPLING_METHODS))}. "
    f"If the upsampling method is 'interpolate', "
    f"the option '--kind' will be used, if given. "
    f"Other upsampling methods that select existing values "
    f"honour the '--tolerance' option. "
    f"Defaults to {DEFAULT_RESAMPLING_METHOD!r}.",
)
@click.option(
    "--frequency",
    "-F",
    help='Temporal aggregation frequency. Use format "<count><offset>" '
    "where <offset> is one of 'H', 'D', 'W', 'M', 'Q', 'Y'. "
    "Use 'all' to aggregate all time steps included in the dataset."
    f"Defaults to {DEFAULT_RESAMPLING_FREQUENCY!r}.",
)
@click.option(
    "--offset",
    "-O",
    help="Offset used to adjust the resampled time labels. Uses same syntax as frequency. "
    "Some Pandas date offset strings are supported as well.",
)
@click.option(
    "--kind",
    "-K",
    type=str,
    default=DEFAULT_INTERPOLATION_KIND,
    help="Interpolation kind which will be used if upsampling method is 'interpolation'. "
    f"May be one of {', '.join(map(repr, INTERPOLATION_KINDS))} where "
    f"{', '.join(map(repr, SPLINE_INTERPOLATION_KINDS))} refer to a spline interpolation of "
    f"zeroth, first, second or third order; 'previous' and 'next' "
    f"simply return the previous or next value of the point. "
    "For more info "
    "refer to scipy.interpolate.interp1d(). "
    f"Defaults to {DEFAULT_INTERPOLATION_KIND!r}.",
)
@click.option(
    "--tolerance",
    "-T",
    type=str,
    help="Tolerance for selective upsampling methods. Uses same syntax as frequency. "
    "If the time delta exceeds the tolerance, "
    "fill values (NaN) will be used. "
    "Defaults to the given frequency.",
)
@cli_option_quiet
@cli_option_verbosity
@click.option(
    "--dry-run",
    default=False,
    is_flag=True,
    help="Just read and process inputs, but don't produce any outputs.",
)
def resample(
    cube,
    config,
    output,
    format,
    variables,
    method,
    frequency,
    offset,
    kind,
    tolerance,
    quiet,
    verbosity,
    dry_run,
):
    """Resample data along the time dimension."""
    configure_cli_output(quiet=quiet, verbosity=verbosity)

    input_path = cube
    config_files = config
    output_path = output
    output_format = format

    from xcube.util.config import load_configs

    config = load_configs(*config_files) if config_files else {}

    if input_path:
        config["input_path"] = input_path
    if output_path:
        config["output_path"] = output_path
    if output_format:
        config["output_format"] = output_format
    if method:
        config["methods"] = method
    if frequency:
        config["frequency"] = frequency
    if offset:
        config["offset"] = offset
    if kind:
        config["interp_kind"] = kind
    if tolerance:
        config["tolerance"] = tolerance
    if variables:
        try:
            variables = set(map(lambda c: str(c).strip(), variables.split(",")))
        except ValueError:
            variables = None
        if variables is not None and next(
            iter(True for var_name in variables if var_name == ""), False
        ):
            variables = None
        if variables is None or len(variables) == 0:
            raise click.ClickException(f"invalid variables {variables!r}")
        config["variables"] = variables

    if "methods" in config:
        methods = config["methods"]
        for method in methods:
            if method not in RESAMPLING_METHODS:
                raise click.ClickException(f"invalid resampling method {method!r}")

    # noinspection PyBroadException
    _resample_in_time(**config, dry_run=dry_run, monitor=LOG.info)

    return 0


def _resample_in_time(
    input_path: str = None,
    variables: Sequence[str] = None,
    metadata: dict[str, Any] = None,
    output_path: str = DEFAULT_OUTPUT_PATH,
    output_format: str = None,
    methods: Sequence[str] = (DEFAULT_RESAMPLING_METHOD,),
    frequency: str = DEFAULT_RESAMPLING_FREQUENCY,
    offset: str = None,
    interp_kind: str = DEFAULT_INTERPOLATION_KIND,
    tolerance: str = None,
    dry_run: bool = False,
    monitor=None,
):
    from xcube.core.dsio import guess_dataset_format, open_cube, write_cube
    from xcube.core.resampling import resample_in_time
    from xcube.core.update import update_dataset_chunk_encoding

    if not output_format:
        output_format = guess_dataset_format(output_path)

    monitor(f"Opening cube from {input_path!r}...")
    with open_cube(input_path) as ds:
        monitor("Resampling...")
        agg_ds = resample_in_time(
            ds,
            frequency=frequency,
            method=methods,
            offset=offset,
            interp_kind=interp_kind,
            tolerance=tolerance,
            time_chunk_size=1,
            var_names=variables,
            metadata=metadata,
        )

        agg_ds = update_dataset_chunk_encoding(
            agg_ds, chunk_sizes={}, format_name=output_format, in_place=True
        )

        monitor(f"Writing resampled cube to {output_path!r}...")
        if not dry_run:
            write_cube(agg_ds, output_path, output_format, cube_asserted=True)

        monitor(f"Done.")
