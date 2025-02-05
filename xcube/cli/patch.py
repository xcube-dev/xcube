# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import warnings
from collections.abc import MutableMapping
from typing import Any, Dict

import click

from xcube.cli.common import (
    cli_option_dry_run,
    cli_option_quiet,
    cli_option_verbosity,
    configure_cli_output,
)
from xcube.constants import LOG

DELETE_ATTR_VALUE = "__delete__"


# noinspection PyShadowingBuiltins
@click.command(name="patch")
@click.argument("DATASET")
@click.option(
    "--metadata",
    "metadata_path",
    metavar="METADATA",
    help="The metadata to be patched."
    " Must be a JSON or YAML file"
    " using Zarr consolidated metadata format.",
)
@click.option(
    "--options",
    "options_path",
    metavar="OPTIONS",
    help="Protocol-specific storage options (see fsspec). Must be a JSON or YAML file.",
)
@cli_option_quiet
@cli_option_verbosity
@cli_option_dry_run
def patch(dataset, metadata_path, options_path, quiet, verbosity, dry_run):
    """
    Patch and consolidate the metadata of a dataset.

    DATASET can be either a local filesystem path or a URL.
    It must point to either a Zarr dataset (*.zarr)
    or a xcube multi-level dataset (*.levels).
    Additional storage options for a given protocol may be passed
    by the OPTIONS option.

    In METADATA, the special attribute value "__delete__" can be used to
    remove that attribute from dataset or array metadata.
    """
    configure_cli_output(quiet=quiet, verbosity=verbosity)
    metadata = load_metadata(metadata_path)
    storage_options = load_storage_options(options_path)
    patch_dataset(dataset, storage_options, metadata, dry_run)


def load_metadata(metadata_path):
    from xcube.util.config import load_json_or_yaml_config

    if not metadata_path:
        raise click.ClickException("Missing metadata to be patched")

    metadata = load_json_or_yaml_config(metadata_path)
    return parse_metadata(metadata)


def parse_metadata(metadata):
    if not isinstance(metadata, dict) or "zarr_consolidated_format" not in metadata:
        raise click.ClickException("Invalid consolidated metadata format")
    zarr_consolidated_format = metadata.get("zarr_consolidated_format")
    if zarr_consolidated_format != 1:
        raise click.ClickException("Unsupported consolidated metadata version")
    metadata = metadata.get("metadata")
    if not isinstance(metadata, dict):
        raise click.ClickException("Invalid metadata format")
    _metadata = dict()
    for k, v in metadata.items():
        if not isinstance(v, dict):
            raise click.ClickException(
                f'Invalid metadata format: entry "{k}" is not an object'
            )
        parts = k.split("/")
        if parts[-1] != ".zattrs":
            warnings.warn(f'Ignoring metadata entry "{k}": can only patch "*/.zattrs"')
        elif len(parts) not in (1, 2):
            warnings.warn(
                f'Ignoring metadata entry {k}: can only patch ".zattrs" of first level'
            )
        else:
            _metadata[k] = v
    if not _metadata:
        raise click.ClickException("No metadata provided")
    return _metadata


def load_storage_options(options_path):
    from xcube.util.config import load_json_or_yaml_config

    storage_options = {}
    if options_path:
        storage_options = load_json_or_yaml_config(options_path)
        if not isinstance(storage_options, dict):
            raise click.ClickException("Invalid storage options format")
    return storage_options


def patch_dataset(
    dataset_path: str,
    storage_options: dict[str, Any],
    metadata: dict[str, Any],
    dry_run: bool,
):
    if dataset_path.endswith(".levels"):
        fs, root = get_fs_and_root(dataset_path, storage_options)
        protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
        prefix = f"{protocol}://" if protocol != "file" else ""
        for item in fs.listdir(root, detail=False):
            if item.endswith(".zarr"):
                _patch_dataset(prefix + item, storage_options, metadata, dry_run)
    else:
        _patch_dataset(dataset_path, storage_options, metadata, dry_run)


def _patch_dataset(
    dataset_path: str,
    storage_options: dict[str, Any],
    metadata: dict[str, Any],
    dry_run: bool,
):
    import zarr

    LOG.info(f"Opening {dataset_path}...")
    zarr_store = _open_zarr_store(dataset_path, storage_options)
    group = zarr.open(zarr_store, mode="r+" if not dry_run else "r")
    for k, v in metadata.items():
        parts = k.split("/")
        item = None
        if len(parts) == 1:
            item = group
        else:
            item_name, _ = parts
            if item_name in group:
                item = group[item_name]
            else:
                warnings.warn(f"Ignoring metadata entry {k}: not found in dataset")
        if item is not None:
            upd_count = 0
            del_count = 0
            for ak, av in v.items():
                if av == DELETE_ATTR_VALUE:
                    if ak in item.attrs:
                        if not dry_run:
                            del item.attrs[ak]
                        del_count += 1
                else:
                    if not dry_run:
                        item.attrs[ak] = av
                    upd_count += 1
            if upd_count and del_count:
                LOG.info(f"{k}: {upd_count} attribute(s) updated, {del_count} deleted")
            elif upd_count:
                LOG.info(f"{k}: {upd_count} attribute(s) updated")
            elif del_count:
                LOG.info(f"{k}: {del_count} attribute(s) deleted")

    if not dry_run:
        zarr.convenience.consolidate_metadata(zarr_store)
    LOG.info(f"Consolidated {dataset_path}")


def _open_zarr_store(
    dataset_path: str, storage_options: dict[str, Any]
) -> MutableMapping[str, bytes]:
    fs, root = get_fs_and_root(dataset_path, storage_options)
    return fs.get_mapper(root=root)


def get_fs_and_root(dataset_path: str, storage_options: dict[str, Any]):
    import fsspec

    protocol, root = fsspec.core.split_protocol(dataset_path)
    protocol = protocol or "file"
    fs = fsspec.filesystem(protocol, **storage_options)
    return fs, root
