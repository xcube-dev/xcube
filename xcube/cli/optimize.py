# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import click

DEFAULT_OUTPUT_PATH = "{input}-optimized.zarr"


# noinspection PyShadowingBuiltins
@click.command(name="optimize")
@click.argument("CUBE")
@click.option(
    "--output",
    "-o",
    metavar="OUTPUT",
    help=f'Output path. The placeholder "{input}" will be replaced by the input\'s filename '
    f'without extension (such as ".zarr"). Defaults to "{DEFAULT_OUTPUT_PATH}".',
    default=DEFAULT_OUTPUT_PATH,
)
@click.option(
    "--in-place",
    "-I",
    "in_place",
    help="Optimize cube in place. Ignores output path.",
    is_flag=True,
)
@click.option(
    "--coords",
    "-C",
    "unchunk_coords",
    help="Also optimize coordinate variables by converting any chunked arrays "
    "into single, non-chunked, contiguous arrays.",
    is_flag=True,
)
def optimize(cube, output, in_place, unchunk_coords):
    """
    Optimize xcube dataset for faster access.

    Reduces the number of metadata and coordinate data files in xcube dataset given by CUBE.
    Consolidated cubes open much faster especially from remote locations, e.g. in object storage,
    because obviously much less HTTP requests are required to fetch initial cube meta
    information. That is, it merges all metadata files into a single top-level JSON file ".zmetadata".
    Optionally, it removes any chunking of coordinate variables
    so they comprise a single binary data file instead of one file per data chunk.
    The primary usage of this command is to optimize data cubes for cloud object storage.
    The command currently works only for data cubes using ZARR format.
    """
    from xcube.core.optimize import optimize_dataset

    optimize_dataset(
        cube, output_path=output, in_place=in_place, unchunk_coords=unchunk_coords
    )
