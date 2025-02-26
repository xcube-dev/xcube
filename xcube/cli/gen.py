# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from collections.abc import Sequence
from typing import List

import click

from xcube.constants import (
    EXTENSION_POINT_DATASET_IOS,
    EXTENSION_POINT_INPUT_PROCESSORS,
    LOG,
    RESAMPLING_METHOD_NAMES,
)
from xcube.core.gen.defaults import DEFAULT_OUTPUT_PATH, DEFAULT_OUTPUT_RESAMPLING
from xcube.util.extension import Extension

resampling_methods = sorted(RESAMPLING_METHOD_NAMES)


# noinspection PyShadowingBuiltins
@click.command(name="gen", context_settings={"ignore_unknown_options": True})
@click.argument("input", nargs=-1)
@click.option(
    "--proc",
    "-P",
    metavar="INPUT-PROCESSOR",
    default=None,
    help=f"Input processor name. "
    f"The available input processor names and additional information about input processors "
    'can be accessed by calling xcube gen --info . Defaults to "default", an input processor '
    'that can deal with simple datasets whose variables have dimensions ("lat", "lon") and '
    "conform with the CF conventions.",
)
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
    help=f"Output path. Defaults to {DEFAULT_OUTPUT_PATH!r}",
)
@click.option(
    "--format",
    "-f",
    metavar="FORMAT",
    help=f"Output format. "
    "Information about output formats can be accessed by calling "
    "xcube gen --info. If omitted, the format will be guessed from the given output path.",
)
@click.option(
    "--size",
    "-S",
    metavar="SIZE",
    help='Output size in pixels using format "<width>,<height>".',
)
@click.option(
    "--region",
    "-R",
    metavar="REGION",
    help='Output region using format "<lon-min>,<lat-min>,<lon-max>,<lat-max>"',
)
@click.option(
    "--variables",
    "--vars",
    metavar="VARIABLES",
    help="Variables to be included in output. "
    'Comma-separated list of names which may contain wildcard characters "*" and "?".',
)
@click.option(
    "--resampling",
    type=click.Choice(resampling_methods),
    default=DEFAULT_OUTPUT_RESAMPLING,
    help="Fallback spatial resampling algorithm to be used for all variables. "
    f"Defaults to {DEFAULT_OUTPUT_RESAMPLING!r}. "
    f"The choices for the resampling algorithm are: {resampling_methods}",
)
@click.option(
    "--append",
    "-a",
    is_flag=True,
    help="Deprecated. The command will now always create, insert, replace, or append input slices.",
)
@click.option(
    "--prof",
    is_flag=True,
    help="Collect profiling information and dump results after processing.",
)
@click.option(
    "--no_sort",
    is_flag=True,
    help="The input file list will not be sorted before creating the xcube dataset. "
    "If --no_sort parameter is passed, the order of the input list will be kept. "
    "This parameter should be used for better performance, "
    "provided that the input file list is in correct order (continuous time).",
)
@click.option(
    "--info",
    "-I",
    is_flag=True,
    help="Displays additional information about format options or about input processors.",
)
@click.option(
    "--dry_run",
    is_flag=True,
    help="Just read and process inputs, but don't produce any outputs.",
)
def gen(
    input: Sequence[str],
    proc: str,
    config: Sequence[str],
    output: str,
    format: str,
    size: str,
    region: str,
    variables: str,
    resampling: str,
    append: bool,
    prof: bool,
    dry_run: bool,
    info: bool,
    no_sort: bool,
):
    """
    Generate xcube dataset.
    Data cubes may be created in one go or successively for all given inputs.
    Each input is expected to provide a single time slice which may be appended, inserted or which may replace an
    existing time slice in the output dataset.
    The input paths may be one or more input files or a pattern that may contain wildcards '?', '*', and '**'.
    The input paths can also be passed as lines of a text file. To do so, provide exactly one input file with
    ".txt" extension which contains the actual input paths to be used.
    """
    dry_run = dry_run
    info_mode = info

    if info_mode:
        print(_format_info())
        return 0

    from xcube.core.gen.config import get_config_dict
    from xcube.core.gen.gen import gen_cube

    config = get_config_dict(
        input_paths=input,
        input_processor_name=proc,
        config_files=config,
        output_path=output,
        output_writer_name=format,
        output_size=size,
        output_region=region,
        output_variables=variables,
        output_resampling=resampling,
        profile_mode=prof,
        append_mode=append,
        no_sort_mode=no_sort,
    )

    gen_cube(dry_run=dry_run, monitor=LOG.info, **config)

    return 0


def _format_info():
    from xcube.util.plugin import get_extension_registry

    iproc_extensions = get_extension_registry().find_extensions(
        EXTENSION_POINT_INPUT_PROCESSORS
    )
    dsio_extensions = get_extension_registry().find_extensions(
        EXTENSION_POINT_DATASET_IOS, lambda e: "w" in e.metadata.get("modes", set())
    )

    help_text = "\nInput processors to be used with option --proc:\n"
    help_text += _format_input_processors(iproc_extensions)
    help_text += (
        '\nFor more input processors use existing "xcube-gen-..." plugins '
        "from the xcube's GitHub organisation or write your own plugin.\n"
    )
    help_text += "\n"
    help_text += "\nOutput formats to be used with option --format:\n"
    help_text += _format_dataset_ios(dsio_extensions)
    help_text += "\n"

    return help_text


def _format_input_processors(input_processors: list[Extension]):
    help_text = ""
    for input_processor in input_processors:
        name = input_processor.name
        description = input_processor.metadata.get("description", "")
        fill = " " * (34 - len(input_processor.name))
        help_text += f"  {name}{fill}{description}\n"
    return help_text


def _format_dataset_ios(dataset_ios: list[Extension]):
    help_text = ""
    for ds_io in dataset_ios:
        name = ds_io.name
        description = ds_io.metadata.get("description", "")
        ext = ds_io.metadata.get("ext", "?")
        fill1 = " " * (24 - len(name))
        fill2 = " " * (10 - len(ext))
        help_text += f"  {name}{fill1}(*.{ext}){fill2}{description}\n"
    return help_text
