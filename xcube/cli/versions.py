# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Optional

import os.path

import click

YAML_FORMAT = "yaml"
JSON_FORMAT = "json"
DEFAULT_FORMAT = YAML_FORMAT


# noinspection PyShadowingBuiltins
@click.command(name="versions")
@click.option(
    "--format",
    "-f",
    "format_name",
    metavar="FORMAT",
    type=click.Choice([YAML_FORMAT, JSON_FORMAT], case_sensitive=False),
    help=f"Output format. "
    f"Must be one of {YAML_FORMAT!r} or {JSON_FORMAT!r}. "
    f"If not given, derived from OUTPUT name extension.",
)
@click.option(
    "--output", "-o", "output_path", metavar="OUTPUT", help=f"Output file path."
)
def versions(format_name: Optional[str], output_path: Optional[str]):
    """
    Get versions of important packages used by xcube.
    """
    from xcube.util.versions import get_xcube_versions

    xcube_versions = get_xcube_versions()

    if not format_name:
        format_name = YAML_FORMAT
        if output_path:
            _, ext = os.path.splitext(output_path)
            if ext.lower() == ".json":
                format_name = JSON_FORMAT

    if format_name.lower() == JSON_FORMAT:
        import json

        text = json.dumps(xcube_versions, indent=2)
    else:
        import yaml

        text = yaml.dump(xcube_versions)

    if output_path:
        with open(output_path, "w") as fp:
            fp.write(text)
    else:
        print(text)
