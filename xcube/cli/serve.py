# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from xcube.cli.common import (
    cli_option_quiet,
    cli_option_verbosity,
    configure_cli_output,
)
from xcube.constants import (
    DEFAULT_SERVER_ADDRESS,
    DEFAULT_SERVER_FRAMEWORK,
    DEFAULT_SERVER_PORT,
)

assets_to_show = ["apis", "endpoints", "openapi", "config", "configschema"]

assets_to_show_choices = list(assets_to_show)
assets_to_show_choices.extend([a + ".yaml" for a in assets_to_show])
assets_to_show_choices.extend([a + ".json" for a in assets_to_show])


@click.command(name="serve")
@click.option(
    "--framework",
    "framework_name",
    metavar="FRAMEWORK",
    default=DEFAULT_SERVER_FRAMEWORK,
    type=click.Choice(["tornado", "flask"]),
    help=f'Web server framework. Defaults to "{DEFAULT_SERVER_FRAMEWORK}"',
)
@click.option(
    "--port",
    "-p",
    metavar="PORT",
    default=None,
    type=int,
    help=f"Service port number. Defaults to {DEFAULT_SERVER_PORT}",
)
@click.option(
    "--address",
    "-a",
    metavar="ADDRESS",
    default=None,
    help=f'Service address. Defaults to "{DEFAULT_SERVER_ADDRESS}".',
)
@click.option(
    "--config",
    "-c",
    "config_paths",
    metavar="CONFIG",
    default=None,
    multiple=True,
    help="Configuration YAML or JSON file. "
    " If multiple are passed,"
    " they will be merged in order."
    " Can be a local filesystem path or an absolute URL.",
)
@click.option(
    "--base-dir",
    "base_dir",
    metavar="BASE_DIR",
    default=None,
    help="Directory used to resolve relative paths"
    " in CONFIG files. Defaults to the parent directory"
    " of (last) CONFIG file."
    " Can be a local filesystem path or an absolute URL.",
)
@click.option(
    "--prefix",
    "url_prefix",
    metavar="URL_PREFIX",
    default=None,
    help="Prefix path to be used for all endpoint URLs."
    ' May include template variables, e.g., "api/{version}".'
    " Can be an absolute URL or a relative URL path.",
)
@click.option(
    "--revprefix",
    "reverse_url_prefix",
    metavar="REVERSE_URL_PREFIX",
    default=None,
    help="Prefix path to be used for reverse endpoint URLs"
    " that may be reported by server responses."
    ' May include template variables, e.g., "/proxy/{port}".'
    " Defaults to value of URL_PREFIX."
    " Can be an absolute URL or relative URL path.",
)
@click.option(
    "--traceperf",
    "trace_perf",
    is_flag=True,
    default=None,
    help="Whether to output extra performance logs.",
)
@click.option(
    "--update-after",
    "update_after",
    metavar="TIME",
    type=float,
    default=None,
    help="Check for server configuration updates every TIME seconds.",
)
@click.option(
    "--stop-after",
    "stop_after",
    metavar="TIME",
    type=float,
    default=None,
    help="Unconditionally stop service after TIME seconds.",
)
@click.option(
    "--show",
    "asset_to_show",
    type=click.Choice(assets_to_show_choices),
    metavar="ASSET",
    nargs=1,
    help=f"Show ASSET and exit. Possible values for ASSET are"
    f" {', '.join(map(repr, assets_to_show))} optionally"
    f" suffixed by '.yaml' or '.json'.",
)
@click.option(
    "--open-viewer",
    "open_viewer",
    is_flag=True,
    help=f"After starting the server, open xcube Viewer in a browser tab.",
)
@cli_option_quiet
@cli_option_verbosity
@click.argument("paths", metavar="[PATHS...]", nargs=-1)
def serve(
    framework_name: str,
    port: int,
    address: str,
    config_paths: list[str],
    base_dir: Optional[str],
    url_prefix: Optional[str],
    reverse_url_prefix: Optional[str],
    trace_perf: Optional[bool],
    update_after: Optional[float],
    stop_after: Optional[float],
    asset_to_show: str,
    open_viewer: bool,
    quiet: bool,
    verbosity: int,
    paths: list[str],
):
    """Run the xcube Server for the given configuration and/or the given
    raster dataset paths given by PATHS.

    Each of the PATHS arguments can point to a raster dataset such as a Zarr
    directory (*.zarr), an xcube multi-level Zarr dataset (*.levels),
    a NetCDF file (*.nc), a GeoTIFF/COG file (*.tiff).

    If one of PATHS is a directory that is not a dataset itself,
    it is scanned for readable raster datasets.

    The --show ASSET option can be used to inspect the current configuration
    of the server. ASSET is one of:

    \b
    apis            outputs the list of APIs provided by the server
    endpoints       outputs the list of all endpoints provided by the server
    openapi         outputs the OpenAPI document representing this server
    config          outputs the effective server configuration
    configschema    outputs the JSON Schema for the server configuration

    The ASSET may be suffixed by ".yaml" or ".json"
    forcing the respective output format. The default format is YAML.

    Note, if --show  is provided, the ASSET will be shown and the program
    will exit immediately.
    """
    from xcube.server.config import normalize_base_dir
    from xcube.server.framework import get_framework_class
    from xcube.server.helpers import ConfigChangeObserver
    from xcube.server.server import Server
    from xcube.util.config import load_configs

    configure_cli_output(quiet=quiet, verbosity=verbosity)

    config = (
        load_configs(*config_paths, exception_type=click.ClickException)
        if config_paths
        else {}
    )

    if port is not None:
        config["port"] = port
    port = config.get("port")
    if port is None:
        port = DEFAULT_SERVER_PORT
        config["port"] = port

    if address is not None:
        config["address"] = address
    address = config.get("address")
    if address is None:
        address = DEFAULT_SERVER_ADDRESS
        config["address"] = address

    if base_dir is not None:
        # Use base_dir CLI option
        pass
    elif "base_dir" in config:
        # Use base_dir from configuration
        base_dir = config["base_dir"]
    elif config_paths:
        # Use base_dir derived from last config file's parent
        config_path = config_paths[-1].replace("\\", "/")
        base_dir = "/".join(config_path.split("/")[:-1])
    else:
        # base_dir is current working directory
        base_dir = str(Path.cwd())
    base_dir = normalize_base_dir(base_dir)
    config["base_dir"] = base_dir

    if url_prefix is not None:
        config["url_prefix"] = url_prefix

    if reverse_url_prefix is not None:
        config["reverse_url_prefix"] = reverse_url_prefix

    if trace_perf is not None:
        config["trace_perf"] = trace_perf

    if paths:
        data_stores: dict[Path, dict[str, Any]] = dict()
        for path in paths:
            if Path(path).exists():
                path = Path(path)
                if path.is_dir():
                    if path.suffix == ".zarr" or path.suffix == ".levels":
                        _add_path_to_data_stores(data_stores, path)
                    else:
                        _add_dir_to_data_stores(data_stores, path)
                else:
                    _add_path_to_data_stores(data_stores, path)
            else:
                # TODO (forman): Support also "s3://" prefixes
                raise click.ClickException(f"File or directory not found: {path}")
        config["DataStores"] = list(data_stores.values())

    framework = get_framework_class(framework_name)()
    server = Server(framework, config)

    if asset_to_show:
        return show_asset(server, asset_to_show)

    if update_after is not None:
        change_observer = ConfigChangeObserver(server, config_paths, update_after)
        server.call_later(update_after, change_observer.check)

    if stop_after is not None:
        server.call_later(stop_after, server.stop)

    if open_viewer:

        def open_xcube_viewer():
            import webbrowser

            server_url = f"http://localhost:{port}"
            webbrowser.open_new_tab(
                f"{server_url}/viewer/"
                f"?serverUrl={server_url}"
                f"&serverId=local"
                f"&serverName=Local"
            )

        server.call_later(2.5, open_xcube_viewer)

    server.start()


def _add_dir_to_data_stores(
    data_stores: dict[Path, dict[str, Any]], dir_path: Path
) -> dict[str, Any]:
    root = dir_path.resolve().absolute()
    if root in data_stores:
        data_store = data_stores[root]
    else:
        data_store = data_stores[root] = dict(
            Identifier=str(root.name), StoreId="file", StoreParams=dict(root=str(root))
        )
    return data_store


def _add_path_to_data_stores(data_stores: dict[Path, dict[str, Any]], path: Path):
    path = path.resolve().absolute()
    data_store = _add_dir_to_data_stores(data_stores, path.parent)
    if "Datasets" in data_store:
        datasets = data_store["Datasets"]
    else:
        datasets = data_store["Datasets"] = []
    datasets.append(dict(Identifier=str(path.name), Path=str(path.name)))


def show_asset(server, asset_to_show: str):
    def to_yaml(obj):
        import yaml

        yaml.safe_dump(obj, sys.stdout, indent=2)

    def to_json(obj):
        import json

        json.dump(obj, sys.stdout, indent=2)

    available_formats = {"yaml": to_yaml, "json": to_json}

    format_name = "yaml"
    splits = asset_to_show.rsplit(".", maxsplit=1)
    if len(splits) == 2:
        asset_to_show, format_name = splits

    if format_name.lower() not in available_formats:
        raise click.ClickException(
            f"Invalid format {format_name}."
            f" Must be one of {', '.join(available_formats.keys())}."
        )
    output_fn = available_formats[format_name.lower()]

    def show_apis():
        output_fn(
            [
                dict(name=api.name, version=api.version, description=api.description)
                for api in server.apis
            ]
        )

    def show_endpoints():
        output_fn(
            [
                route.path
                for api in server.apis
                for route in (api.routes + api.static_routes)
            ]
        )

    def show_open_api():
        output_fn(server.ctx.get_open_api_doc())

    def show_config():
        output_fn(server.ctx.config.defrost())

    def show_config_schema():
        output_fn(server.config_schema.to_dict())

    available_commands = {
        "apis": show_apis,
        "endpoints": show_endpoints,
        "openapi": show_open_api,
        "config": show_config,
        "configschema": show_config_schema,
    }

    command_fn = available_commands.get(asset_to_show)
    if command_fn is None:
        raise click.ClickException(
            f"Invalid command {asset_to_show}. "
            f"Possible commands are "
            f"{', '.join(' '.join(k) for k in available_commands.keys())}."
        )
    return command_fn()
