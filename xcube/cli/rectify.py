# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import functools
import operator
from collections.abc import Sequence
from typing import Optional, Tuple

import click

from xcube.cli.common import (
    assert_positive_int_item,
    cli_option_quiet,
    cli_option_verbosity,
    configure_cli_output,
    parse_cli_sequence,
)
from xcube.constants import FORMAT_NAME_MEM, FORMAT_NAME_NETCDF4, FORMAT_NAME_ZARR, LOG

OUTPUT_FORMAT_NAMES = [FORMAT_NAME_ZARR, FORMAT_NAME_NETCDF4, FORMAT_NAME_MEM]

DEFAULT_OUTPUT_PATH = "out.zarr"
DEFAULT_XY_NAMES = "lon,lat"
DEFAULT_DELTA = 1e-5
DEFAULT_DRY_RUN = False
DEFAULT_CRS = "EPSG:4326"


# noinspection PyShadowingBuiltins
@click.command(name="rectify", hidden=True)
@click.argument("dataset", metavar="INPUT")
@click.option(
    "--xy-vars",
    "xy_var_names",
    help=f"Comma-separated names of variables providing x any y coordinates. "
    f'If omitted, names will be guessed from available coordinate variables in INPUT, e.g. "lon,lat".',
)
@click.option(
    "--var",
    "-v",
    "var_names",
    multiple=True,
    metavar="VARIABLES",
    help="Comma-separated list of names of variables to be included or multiple options may be given. "
    "If omitted, all variables in INPUT will be reprojected.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    metavar="OUTPUT",
    default=DEFAULT_OUTPUT_PATH,
    help=f"Output path. Defaults to {DEFAULT_OUTPUT_PATH!r}.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    metavar="FORMAT",
    type=click.Choice(OUTPUT_FORMAT_NAMES),
    help="Output format. If omitted, format will be guessed from OUTPUT.",
)
@click.option(
    "--size",
    "-s",
    "output_size",
    metavar="SIZE",
    help='Output size in pixels using format "WIDTH,HEIGHT", e.g. "2048,1024". '
    "If omitted, a size will be computed so spatial resolution of INPUT is preserved.",
)
@click.option(
    "--tile_size",
    "-t",
    "output_tile_size",
    metavar="TILE_SIZE",
    help='Output tile size in pixels using format "WIDTH,HEIGHT", e.g. "512,512". '
    "If given, the output will be chunked w.r.t. TILE_SIZE. Otherwise the output will NOT be chunked.",
)
@click.option(
    "--point",
    "-p",
    "output_point",
    metavar="POINT",
    help="Output spatial coordinates of the point referring to pixel col=0.5,row=0.5 "
    'using format "LON,LAT" or "X,Y", e.g. "1.2,53.5". '
    "If omitted, the default reference point will the INPUT's minimum spatial coordinates.",
)
@click.option(
    "--res",
    "-r",
    "output_res",
    metavar="RES",
    type=float,
    help="Output spatial resolution in decimal degrees. "
    "If omitted, the default resolution will be close to the spatial resolution of INPUT.",
)
@click.option(
    "--crs",
    "-r",
    "output_crs",
    metavar="CRS",
    help="Output spatial coordinate reference system (CRS). "
    f'If omitted, the default CRS will be "{DEFAULT_CRS}".',
)
@click.option(
    "--delta",
    "-d",
    type=float,
    default=DEFAULT_DELTA,
    help="Relative maximum delta for detection whether a "
    "target pixel center is within a source pixel's boundary.",
)
@cli_option_quiet
@cli_option_verbosity
@click.option(
    "--dry-run",
    default=DEFAULT_DRY_RUN,
    is_flag=True,
    help="Just read and process INPUT, but don't produce any outputs.",
)
def rectify(
    dataset: str,
    xy_var_names: str = None,
    var_names: str = None,
    output_path: str = None,
    output_format: str = None,
    output_size: str = None,
    output_tile_size: str = None,
    output_point: str = None,
    output_res: float = None,
    output_crs: str = None,
    delta: float = DEFAULT_DELTA,
    quiet: bool = False,
    verbosity: int = 0,
    dry_run: bool = DEFAULT_DRY_RUN,
):
    """Rectify a dataset to WGS-84 using its per-pixel geo-locations."""
    configure_cli_output(quiet=quiet, verbosity=verbosity)

    input_path = dataset

    xy_var_names = parse_cli_sequence(
        xy_var_names, metavar="VARIABLES", num_items=2, item_plural_name="names"
    )
    var_name_lists = [
        parse_cli_sequence(
            var_name_specifier, metavar="VARIABLES", item_plural_name="names"
        )
        for var_name_specifier in var_names
    ]
    var_name_flat_list = functools.reduce(operator.iconcat, var_name_lists, [])

    output_size = parse_cli_sequence(
        output_size,
        metavar="SIZE",
        num_items=2,
        item_plural_name="sizes",
        item_parser=int,
        item_validator=assert_positive_int_item,
    )
    output_tile_size = parse_cli_sequence(
        output_tile_size,
        metavar="TILE_SIZE",
        num_items=2,
        item_plural_name="tile sizes",
        item_parser=int,
        item_validator=assert_positive_int_item,
    )
    output_point = parse_cli_sequence(
        output_point,
        metavar="POINT",
        num_items=2,
        item_plural_name="coordinates",
        item_parser=float,
    )

    # noinspection PyBroadException
    _rectify(
        input_path,
        xy_var_names,
        None if len(var_name_flat_list) == 0 else var_name_flat_list,
        output_path,
        output_format,
        output_size,
        output_tile_size,
        output_point,
        output_res,
        output_crs,
        delta,
        dry_run=dry_run,
        monitor=LOG.info,
    )

    return 0


def _rectify(
    input_path: str,
    xy_names: Optional[tuple[str, str]],
    var_names: Optional[Sequence[str]],
    output_path: str,
    output_format: Optional[str],
    output_size: Optional[tuple[int, int]],
    output_tile_size: Optional[tuple[int, int]],
    output_point: Optional[tuple[float, float]],
    output_res: Optional[float],
    output_crs: Optional[str],
    delta: float,
    dry_run: bool,
    monitor,
):
    import pyproj.crs

    from xcube.core.dsio import guess_dataset_format, open_dataset, write_dataset
    from xcube.core.gridmapping import GridMapping
    from xcube.core.resampling import rectify_dataset
    from xcube.core.sentinel3 import is_sentinel3_product, open_sentinel3_product

    if not output_format:
        output_format = guess_dataset_format(output_path)

    output_gm = None
    output_gm_given = (
        output_size is not None,
        output_point is not None,
        output_res is not None,
        output_crs is not None,
    )
    if all(output_gm_given):
        output_gm = GridMapping.regular(
            size=output_size,
            xy_min=output_point,
            xy_res=output_res,
            crs=pyproj.crs.CRS.from_user_input(output_crs),
        )
    elif any(output_gm_given):
        raise click.ClickException(
            "SIZE, POINT, RES, and CRS must all be given or none of them."
        )

    monitor(f"Opening dataset from {input_path!r}...")

    if is_sentinel3_product(input_path):
        src_ds = open_sentinel3_product(input_path)
    else:
        src_ds = open_dataset(input_path)

    monitor("Rectifying...")
    rectified_ds = rectify_dataset(
        src_ds,
        xy_var_names=xy_names,
        var_names=var_names,
        target_gm=output_gm,
        tile_size=output_tile_size,
        uv_delta=delta,
    )

    if rectified_ds is None:
        monitor(
            f"Dataset {input_path} does not seem to have an intersection with bounding box"
        )
        return

    monitor(f"Writing rectified dataset to {output_path!r}...")
    if not dry_run:
        write_dataset(rectified_ds, output_path, output_format)

    monitor(f"Done.")
