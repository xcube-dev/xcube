# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import sys

import click


# noinspection PyShadowingBuiltins
@click.command(name="extract")
@click.argument("cube")
@click.argument("points")
@click.option(
    "--output",
    "-o",
    metavar="OUTPUT",
    help="Output path. If omitted, output is written to stdout.",
)
@click.option(
    "--format",
    "-f",
    metavar="FORMAT",
    type=click.Choice(["csv", "json", "xlsx"]),
    help="Output format. Currently, only 'csv' is supported.",
    default="csv",
)
@click.option(
    "--coords", "-C", is_flag=True, help="Include cube cell coordinates in output."
)
@click.option(
    "--bounds",
    "-B",
    is_flag=True,
    help="Include cube cell coordinate boundaries (if any) in output.",
)
@click.option(
    "--indexes", "-I", is_flag=True, help="Include cube cell indexes in output."
)
@click.option(
    "--refs", "-R", is_flag=True, help="Include point values as reference in output."
)
def extract(
    cube,
    points,
    output=None,
    format=None,
    coords=False,
    bounds=False,
    indexes=False,
    refs=False,
):
    """Extract cube points.

    Extracts data cells from CUBE at coordinates given in each POINTS record and writes the resulting values to given
    output path and format.

    POINTS must be a CSV file that provides at least the columns "lon", "lat", and "time". The "lon" and "lat"
    columns provide a point's location in decimal degrees. The "time" column provides a point's date or
    date-time. Its format should preferably be ISO, but other formats may work as well.
    """
    if format != "csv":
        raise click.ClickException(f"Format {format!r} is not supported.")

    import pandas as pd

    cube_path = cube
    points_path = points
    output_path = output
    include_coords = coords
    include_bounds = bounds
    include_indexes = indexes
    include_refs = refs

    from xcube.core.dsio import open_dataset
    from xcube.core.extract import (
        get_cube_values_for_points,
        DEFAULT_INDEX_NAME_PATTERN,
        DEFAULT_REF_NAME_PATTERN,
    )

    # We may make the following CLI options
    index_name_pattern = DEFAULT_INDEX_NAME_PATTERN
    ref_name_pattern = DEFAULT_REF_NAME_PATTERN
    time_col_names = ["time"]

    points = pd.read_csv(
        points_path, parse_dates=time_col_names
    )
    points.time = points.time.dt.tz_localize(tz=None)
    with open_dataset(cube_path) as cube:
        values = get_cube_values_for_points(
            cube,
            points,
            include_coords=include_coords,
            include_bounds=include_bounds,
            include_indexes=include_indexes,
            index_name_pattern=index_name_pattern,
            include_refs=include_refs,
            ref_name_pattern=ref_name_pattern,
        ).to_dataframe()
        values.to_csv(
            output_path if output_path else sys.stdout,
            # We may make the following CLI options
            sep=",",
            date_format="%Y-%m-%dT%H:%M:%SZ",
            index=True,
        )
