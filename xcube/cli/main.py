# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import sys

import click

from xcube.cli.common import (
    cli_option_scheduler,
    cli_option_traceback,
    handle_cli_exception,
    new_cli_ctx_obj,
    configure_logging,
    configure_warnings,
)
from xcube.constants import EXTENSION_POINT_CLI_COMMANDS, LOG_LEVELS, LOG_LEVEL_OFF_NAME
from xcube.util.plugin import get_extension_registry
from xcube.version import version


# noinspection PyShadowingBuiltins,PyUnusedLocal
@click.group(name="xcube")
@click.version_option(version)
@cli_option_traceback
@cli_option_scheduler
@click.option(
    "--loglevel",
    "log_level",
    metavar="LOG_LEVEL",
    type=click.Choice(LOG_LEVELS),
    default=LOG_LEVEL_OFF_NAME,
    help=f"Log level."
    f' Must be one of {", ".join(LOG_LEVELS)}.'
    f" Defaults to {LOG_LEVEL_OFF_NAME}."
    f" If the level is not {LOG_LEVEL_OFF_NAME},"
    f" any log messages up to the given level will be"
    f" written either to the console (stderr)"
    f" or LOG_FILE, if provided.",
)
@click.option(
    "--logfile",
    "log_file",
    metavar="LOG_FILE",
    help=f"Log file path."
    f" If given, any log messages will redirected into"
    f" LOG_FILE. Disables console output"
    f" unless otherwise enabled, e.g.,"
    f" using the --verbose flag."
    f" Effective only if LOG_LEVEL"
    f" is not {LOG_LEVEL_OFF_NAME}.",
)
@click.option(
    "--warnings",
    "-w",
    is_flag=True,
    help="Show any warnings emitted during operation"
    " (warnings are hidden by default).",
)
def cli(traceback=False, scheduler=None, log_level=None, log_file=None, warnings=None):
    """xcube Toolkit"""
    configure_logging(log_file=log_file, log_level=log_level)
    configure_warnings(warnings)


_cli_commands_registered = False


def _register_cli_commands():
    global _cli_commands_registered
    if _cli_commands_registered:
        return
    _cli_commands_registered = True
    # Add registered CLI commands
    registry = get_extension_registry()
    for command in registry.find_components(EXTENSION_POINT_CLI_COMMANDS):
        cli.add_command(command)


_register_cli_commands()


def main(args=None):
    # noinspection PyBroadException
    ctx_obj = new_cli_ctx_obj()
    try:
        exit_code = cli.main(args=args, obj=ctx_obj, standalone_mode=False)
    except Exception as e:
        exit_code = handle_cli_exception(
            e, traceback_mode=ctx_obj.get("traceback", False)
        )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
