# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import json
import sys
from collections.abc import Sequence
from typing import AbstractSet, Any, Dict, List, Optional, Set

import click

from xcube.constants import (
    EXTENSION_POINT_DATA_OPENERS,
    EXTENSION_POINT_DATA_STORES,
    EXTENSION_POINT_DATA_WRITERS,
    LOG,
)
from xcube.util.plugin import get_extension_registry

_NO_TITLE = "<no title>"
_NO_DESCRIPTION = "<no description>"
_UNKNOWN_EXTENSION = "<unknown!>"


@click.command(name="list")
def store_list():
    """List names of data stores."""
    print(f"Data stores:")
    count = _dump_data_stores()
    print(f"{count} data store{'s' if count != 1 else ''} found.")


@click.command(name="list")
def opener_list():
    """List names of data openers."""
    print(f"Data openers:")
    count = _dump_data_openers()
    print(f"{count} data opener{'s' if count != 1 else ''} found.")


@click.command(name="list")
def writer_list():
    """List names of data writers."""
    print(f"Data writers:")
    count = _dump_data_writers()
    print(f"{count} data writer{'s' if count != 1 else ''} found.")


@click.command(name="info")
@click.argument("store_id", metavar="STORE")
@click.argument("store_params", metavar="PARAMS", nargs=-1)
@click.option(
    "-P", "--params", "show_params", is_flag=True, help="Show data store parameters."
)
@click.option(
    "-O",
    "--openers",
    "show_openers",
    is_flag=True,
    help="Show available data store openers.",
)
@click.option(
    "-W",
    "--writers",
    "show_writers",
    is_flag=True,
    help="Show available data store writers.",
)
@click.option(
    "-D", "--data", "show_data_ids", is_flag=True, help="Show available data resources."
)
@click.option(
    "-j", "--json", "use_json_format", is_flag=True, help="Output using JSON format."
)
def store_info(
    store_id: str,
    store_params: list[str],
    show_params: bool,
    show_openers: bool,
    show_writers: bool,
    show_data_ids: bool,
    use_json_format: bool,
):
    """Show data store information.

    Dumps detailed data store information in human readable form or as JSON, when using the --json option.

    You can obtain valid STORE names using command "xcube store list".

    Note some stores require provision of parameters PARAMS
    when using one of the options --openers/-O, --writers/-W, or --data/-D.
    To find out which parameters are available use the command with just the --params/-P option first.
    """
    extension = get_extension_registry().get_extension(
        EXTENSION_POINT_DATA_STORES, store_id
    )
    from xcube.core.store import (
        DataStoreError,
        MutableDataStore,
        get_data_store_params_schema,
    )

    params_schema = get_data_store_params_schema(store_id)
    description = extension.metadata.get("description")
    requires_store_instance = any((show_openers, show_writers, show_data_ids))
    data_store = None
    try:
        if requires_store_instance:
            data_store = _new_data_store(store_id, store_params)
    except DataStoreError as e:
        msg = f"Failed to instantiate data store {store_id!r}"
        if store_params:
            msg += f" with parameters {store_params!r}"
        msg += f": {e}"
        raise click.ClickException(msg) from e
    if use_json_format:
        d = dict()
        d["store_id"] = store_id
        if description:
            d["description"] = description
        if show_params:
            d["params_schema"] = params_schema.to_dict()
        if show_openers:
            d["opener_ids"] = data_store.get_data_opener_ids()
        if show_writers and isinstance(data_store, MutableDataStore):
            d["writer_ids"] = data_store.get_data_writer_ids()
        if show_data_ids:
            d["data_ids"] = sorted(data_store.get_data_ids())
        if show_openers:
            print(json.dumps(d, indent=2))
    else:
        print(f"\nData store description:")
        print(f"  {description or _NO_DESCRIPTION}")
        if show_params:
            print(_format_params_schema(params_schema))
        if show_openers:
            print(f"\nData openers:")
            _dump_store_openers(data_store)
        if show_writers:
            if isinstance(data_store, MutableDataStore):
                print(f"\nData writers:")
                _dump_store_writers(data_store)
            else:
                print(
                    f'No writers available, because data store "{store_id}" is not mutable.'
                )
        if show_data_ids:
            print(f"\nData resources:")
            count = _dump_store_data_ids(data_store)
            print(f"{count} data resource{'s' if count != 1 else ''} found.")


@click.command(name="data")
@click.argument("store_id", metavar="STORE")
@click.argument("data_id", metavar="DATA")
@click.argument("store_params", metavar="PARAMS", nargs=-1)
def store_data(store_id: str, data_id: str, store_params: list[str]):
    """Show data resource information.

    Show the data descriptor for data resource DATA in data store STORE.
    Note some stores require provision of store parameters PARAMS.
    Use "xcube io store info STORE -P" command to find out which parameters are available/required.
    """
    data_store = _new_data_store(store_id, store_params)
    data_descriptor = data_store.describe_data(data_id)
    print(f'Descriptor for data resource "{data_id}" in data store "{store_id}":')
    print(json.dumps(data_descriptor.to_dict(), indent=2))


@click.command(name="info")
@click.argument("opener_id", metavar="OPENER")
def opener_info(opener_id: str):
    """Show data opener information.
    You can obtain valid OPENER names using command "xcube io opener list".
    """
    extension = get_extension_registry().get_extension(
        EXTENSION_POINT_DATA_OPENERS, opener_id
    )
    description = extension.metadata.get("description")
    if description:
        print(description)
    from xcube.core.store import new_data_opener

    opener_ = new_data_opener(opener_id)
    params_schema = opener_.get_open_data_params_schema()
    print(_format_params_schema(params_schema))


@click.command(name="info")
@click.argument("writer_id", metavar="WRITER")
def writer_info(writer_id: str):
    """Show data opener information.
    You can obtain valid WRITER names using command "xcube io writer list".
    """
    extension = get_extension_registry().get_extension(
        EXTENSION_POINT_DATA_WRITERS, writer_id
    )
    description = extension.metadata.get("description")
    if description:
        print(description)
    from xcube.core.store import new_data_writer

    writer_ = new_data_writer(writer_id)
    params_schema = writer_.get_write_data_params_schema()
    print(_format_params_schema(params_schema))


_DEFAULT_DUMP_OUTPUT = "store-dump.json"
_SHORT_INCLUDE = ",".join(
    [
        "store.store_instance_id",
        "store.title",
        "data.data_id",
        "data.bbox",
        "data.spatial_res",
        "data.time_range",
        "data.time_period",
        "var.name",
        "var.dtype",
        "var.dims",
    ]
)


@click.command(name="dump")
@click.option(
    "-o",
    "--output",
    "output_file_path",
    metavar="OUTPUT",
    default=None,
    help=f"Output filename. Output will be written as JSON."
    f' Defaults to "{_DEFAULT_DUMP_OUTPUT}".',
)
@click.option(
    "-c",
    "--config",
    "config_file_path",
    metavar="CONFIG",
    help="Store configuration filename. May use JSON or YAML format.",
)
@click.option(
    "-t",
    "--type",
    "data_type",
    metavar="TYPE",
    help="Data type. If given, only data resources that satisfy the "
    'data type are listed. E.g. "dataset" or "geodataframe"',
)
@click.option(
    "-S",
    "--short",
    "short_form",
    is_flag=True,
    help=f'Short form. Forces option "--includes={_SHORT_INCLUDE}".',
)
@click.option(
    "-I",
    "--includes",
    "include_props",
    metavar="INCLUDE_LIST",
    help="Comma-separated list of properties to be included"
    ' from stores (prefix "store."),'
    ' data resources (prefix "data.") of stores,'
    ' and variables (prefix "var.") of data resources.',
)
@click.option(
    "-E",
    "--excludes",
    "exclude_props",
    metavar="EXCLUDE_LIST",
    help="Comma-separated list of properties to be excluded"
    ' from stores (prefix "store."),'
    ' data resources (prefix "data.") of stores,'
    ' and variables (prefix "var.") of data resources.',
)
@click.option("--csv", "csv_format", is_flag=True, help=f"Use CSV output format.")
@click.option("--yaml", "yaml_format", is_flag=True, help=f"Use YAML output format.")
@click.option(
    "--json", "json_format", is_flag=True, help=f"Use JSON output format (the default)."
)
def dump(
    output_file_path: Optional[str],
    config_file_path: Optional[str],
    data_type: Optional[str],
    short_form: bool,
    include_props: str,
    exclude_props: str,
    csv_format: bool,
    yaml_format: bool,
    json_format: bool,
):
    """Dump metadata of given data stores.

    Dumps data store metadata and metadata for a store's data resources
    for given data stores  into a JSON file.
    Data stores may be selected and configured by a configuration file CONFIG,
    which may have JSON or YAML format.
    For example, this YAML configuration configures a single directory data store:

    \b
    this_dir:
        title: Current Dir
        description: A store that represents my current directory
        store_id: "directory"
        store_params:
            base_dir: "."

    """
    import json
    import os.path

    import yaml

    from xcube.core.store import DataStoreConfig, DataStorePool

    if csv_format:
        output_format = "csv"
        ext = ".csv"
    elif yaml_format:
        output_format = "yaml"
        ext = ".yml"
    elif json_format:
        output_format = "json"
        ext = ".json"
    elif output_file_path is not None:
        path_no_ext, ext = os.path.splitext(output_file_path)
        if ext in (".csv", ".txt"):
            output_format = "csv"
        elif ext in (".yaml", ".yml"):
            output_format = "yaml"
        else:
            output_format = "json"
    else:
        output_format = "json"
        ext = ".json"

    if output_file_path is None:
        path_no_ext, _ = os.path.splitext(_DEFAULT_DUMP_OUTPUT)
        output_file_path = path_no_ext + ext

    include_props = _parse_props(include_props) if include_props else None
    exclude_props = _parse_props(exclude_props) if exclude_props else None

    if short_form:
        short_include_props = _parse_props(_SHORT_INCLUDE)
        include_props = include_props or {}
        for data_key in ("store", "data", "var"):
            include_props[data_key] = include_props.get(data_key, set()).union(
                short_include_props[data_key]
            )

    if config_file_path:
        store_pool = DataStorePool.from_file(config_file_path)
    else:
        extensions = get_extension_registry().find_extensions(
            EXTENSION_POINT_DATA_STORES
        )
        store_configs = {
            extension.name: DataStoreConfig(
                extension.name,
                title=extension.metadata.get("title"),
                description=extension.metadata.get("description"),
            )
            for extension in extensions
            if extension.name not in ("memory", "directory", "s3")
        }
        store_pool = DataStorePool(store_configs)

    dump_data = _get_store_data_var_tuples(
        store_pool, data_type, include_props, exclude_props
    )

    if output_format == "csv":
        column_names = None
        column_names_set = None
        rows = []
        for store_dict, data_dict, var_dict in dump_data:
            if store_dict is None:
                break
            row = {}
            row.update({"store." + k: v for k, v in store_dict.items()})
            row.update({"data." + k: v for k, v in data_dict.items()})
            row.update({"var." + k: v for k, v in var_dict.items()})
            rows.append(row)
            if column_names_set is None:
                column_names = list(row.keys())
                column_names_set = set(column_names)
            else:
                for k in row.keys():
                    if k not in column_names_set:
                        column_names.append(k)
                        column_names_set.add(k)

        def format_cell_value(value: Any) -> str:
            return str(value) if value is not None else ""

        sep = "\t"
        with open(output_file_path, "w") as fp:
            if column_names:
                fp.write(sep.join(column_names) + "\n")
                for row in rows:
                    fp.write(
                        sep.join(
                            map(
                                format_cell_value,
                                tuple(row.get(k) for k in column_names),
                            )
                        )
                        + "\n"
                    )

        LOG.info(f"Dumped {len(rows)} store entry/ies to {output_file_path}.")

    else:
        last_store_dict = None
        last_data_dict = None
        vars_list = []
        data_list = []
        store_list = []
        for store_dict, data_dict, var_dict in dump_data:
            if data_dict is not last_data_dict or data_dict is None:
                if last_data_dict is not None:
                    last_data_dict["data_vars"] = vars_list
                    vars_list = []
                    data_list.append(last_data_dict)
                last_data_dict = data_dict
            if store_dict is not last_store_dict or store_dict is None:
                if last_store_dict is not None:
                    last_store_dict["data"] = data_list
                    data_list = []
                    store_list.append(last_store_dict)
                last_store_dict = store_dict
            if var_dict:
                vars_list.append(var_dict)

        with open(output_file_path, "w") as fp:
            if output_format == "json":
                from xcube.util.jsonencoder import NumpyJSONEncoder

                json.dump(dict(stores=store_list), fp, indent=2, cls=NumpyJSONEncoder)
            else:
                yaml.dump(dict(stores=store_list), fp, indent=2)

        LOG.info(f"Dumped entries of {len(store_list)} store(s) to {output_file_path}.")


def _get_store_data_var_tuples(store_pool, data_type, include_props, exclude_props):
    import time

    for store_instance_id in store_pool.store_instance_ids:
        t0 = time.perf_counter()
        print(f'Generating entries for store "{store_instance_id}"...')

        store_config = store_pool.get_store_config(store_instance_id)
        store_dict = dict(
            store_instance_id=store_instance_id,
            store_id=store_instance_id,
            title=store_config.title,
            description=store_config.description,
            data_type=data_type,
            data=[],
        )
        store_dict = _filter_dict(store_dict, "store", include_props, exclude_props)

        if "data" not in store_dict:
            yield store_dict, {}, {}
        else:
            del store_dict["data"]

            try:
                store_instance = store_pool.get_store(store_instance_id)
            except BaseException as error:
                LOG.error(f'Cannot open store "{store_instance_id}": {error}')
                continue

            try:
                data_descriptors = store_instance.search_data(data_type=data_type)
            except BaseException as error:
                LOG.error(f'Cannot search store "{store_instance_id}": {error}')
                continue

            for data_descriptor in data_descriptors:
                print(f'Processing data resource "{data_descriptor.data_id}"...')
                data_dict = data_descriptor.to_dict()
                data_dict = _filter_dict(
                    data_dict, "data", include_props, exclude_props
                )
                if "data_vars" not in data_dict:
                    yield store_dict, data_dict, {}
                else:
                    var_dicts = data_dict.pop("data_vars")
                    for var_name, var_dict in var_dicts.items():
                        var_dict["name"] = var_name
                        var_dict = _filter_dict(
                            var_dict, "var", include_props, exclude_props
                        )
                        yield store_dict, data_dict, var_dict

        print(
            f'Done generating entries for store "{store_instance_id}" after '
            + f"{time.perf_counter() - t0:.2f} seconds"
        )

    # yield Terminator
    yield None, None, None


def _filter_dict(
    data: dict[str, Any],
    selector: str,
    include_props: dict[str, set[str]] = None,
    exclude_props: dict[str, set[str]] = None,
) -> dict[str, Any]:
    includes = (
        (include_props.get(selector) or None) if include_props is not None else None
    )
    excludes = (
        (exclude_props.get(selector) or None) if exclude_props is not None else None
    )
    if includes is None and excludes is None:
        return data
    return {
        key: value
        for key, value in data.items()
        if (includes is None or key in includes)
        and (excludes is None or key not in excludes)
    }


def _parse_props(props: str) -> dict[str, AbstractSet]:
    parsed_props = dict(store=set(), data=set(), var=set())
    for p in props.split(","):
        try:
            prefix, name = p.strip().split(".")
            parsed_props[prefix].add(name)
        except (ValueError, KeyError):
            raise click.ClickException(f"Invalid include/exclude property: {p}")
    if parsed_props["var"]:
        parsed_props["data"].add("data_vars")
    if parsed_props["data"]:
        parsed_props["store"].add("data")
    return parsed_props


@click.group()
def store():
    """Tools for xcube's data stores."""
    pass


@click.group()
def opener():
    """Tools for xcube's data openers."""
    pass


@click.group()
def writer():
    """Tools for xcube's data writers."""
    pass


store.add_command(store_list)
store.add_command(store_info)
store.add_command(store_data)
opener.add_command(opener_list)
opener.add_command(opener_info)
writer.add_command(writer_list)
writer.add_command(writer_info)


@click.group(hidden=True)
def io():
    """Tools for xcube's generic I/O system."""
    pass


io.add_command(store)
io.add_command(opener)
io.add_command(writer)
io.add_command(dump)


# from xcube.util.jsonschema import JsonObjectSchema


# noinspection PyUnresolvedReferences
def _format_params_schema(
    params_schema: "xcube.util.jsonschema.JsonObjectSchema",
) -> str:
    text = []
    if params_schema.properties:
        text.append("Parameters:")
        for param_name, param_schema in params_schema.properties.items():
            text.append(
                f"{'* ' if param_name in params_schema.required else '  '}"
                f"{param_name:>24s}  {_format_param_schema(param_schema)}"
            )
    else:
        text.append(f"No parameters required.")
    return "\n".join(text)


# noinspection PyUnresolvedReferences
def _format_required_params_schema(
    params_schema: "xcube.util.jsonschema.JsonObjectSchema",
) -> str:
    text = ["Required parameters:"]
    for param_name, param_schema in params_schema.properties.items():
        if param_name in params_schema.required:
            text.append(f"  {param_name:>24s}  {_format_param_schema(param_schema)}")
    return "\n".join(text)


# noinspection PyUnresolvedReferences
def _format_param_schema(param_schema: "xcube.util.jsonschema.JsonSchema"):
    from xcube.util.undefined import UNDEFINED

    param_info = []
    if param_schema.title:
        param_info.append(
            param_schema.title + ("" if param_schema.title.endswith(".") else ".")
        )
    if param_schema.description:
        param_info.append(
            param_schema.description
            + ("" if param_schema.description.endswith(".") else ".")
        )
    if param_schema.enum:
        param_info.append(
            f"Must be one of {', '.join(map(json.dumps, param_schema.enum))}."
        )
    if param_schema.const != UNDEFINED:
        param_info.append(f"Must be {json.dumps(param_schema.const)}.")
    if param_schema.default != UNDEFINED:
        param_info.append(f"Defaults to {json.dumps(param_schema.default)}.")
    param_info_text = " " + " ".join(param_info) if param_info else ""
    return f"({param_schema.type}){param_info_text}"


def _dump_data_stores() -> int:
    return _dump_extensions(EXTENSION_POINT_DATA_STORES)


def _dump_data_openers() -> int:
    return _dump_extensions(EXTENSION_POINT_DATA_OPENERS)


def _dump_data_writers() -> int:
    return _dump_extensions(EXTENSION_POINT_DATA_WRITERS)


def _dump_extensions(point: str) -> int:
    count = 0
    for extension in get_extension_registry().find_extensions(point):
        print(
            f"  {extension.name:>24s}  {extension.metadata.get('description', '<no description>')}"
        )
        count += 1
    return count


# noinspection PyUnresolvedReferences
def _dump_store_openers(
    data_store: "xcube.core.store.DataStore", data_id: str = None
) -> int:
    return _dump_named_extensions(
        EXTENSION_POINT_DATA_OPENERS, data_store.get_data_opener_ids(data_id=data_id)
    )


# noinspection PyUnresolvedReferences
def _dump_store_writers(data_store: "xcube.core.store.DataStore") -> int:
    return _dump_named_extensions(
        EXTENSION_POINT_DATA_WRITERS, data_store.get_data_writer_ids()
    )


# noinspection PyUnresolvedReferences
def _dump_store_data_ids(data_store: "xcube.core.store.DataStore") -> int:
    count = 0
    for data_id, data_attrs in sorted(data_store.get_data_ids(include_attrs=["title"])):
        print(f"  {data_id:>32s}  {data_attrs.get('title') or _NO_TITLE}")
        count += 1
    return count


def _dump_named_extensions(point: str, names: Sequence[str]) -> int:
    count = 0
    for name in names:
        extension = get_extension_registry().get_extension(point, name)
        if extension:
            print(
                f"  {name:>24s}  {extension.metadata.get('description', _NO_DESCRIPTION)}"
            )
        else:
            print(f"  {name:>24s}  {_UNKNOWN_EXTENSION}")
        count += 1
    return count


# noinspection PyUnresolvedReferences
def _dump_data_resources(data_store: "xcube.core.store.DataStore") -> int:
    count = 0
    for data_id, title in data_store.get_data_ids():
        print(f"  {data_id:<32s}  {title or _NO_TITLE}")
        count += 1
    return count


# noinspection PyUnresolvedReferences
def _new_data_store(
    store_id: str, store_params: list[str]
) -> "xcube.core.store.DataStore":
    from xcube.core.store import get_data_store_params_schema, new_data_store

    store_params_dict = dict()
    if store_params:
        for p_assignment in store_params:
            p_name_value = p_assignment.split("=", maxsplit=2)
            p_name = p_name_value[0].strip()
            if not p_name:
                raise click.ClickException(
                    f"Invalid parameter assignment: {p_assignment}"
                )
            if len(p_name_value) == 2:
                # Passed as name=value
                p_value = p_name_value[1].strip()
                try:
                    p_value = json.loads(p_value)
                except json.decoder.JSONDecodeError:
                    pass
            else:
                # Passed name as flag
                p_value = True
            store_params_dict[p_name] = p_value
    return new_data_store(store_id, **store_params_dict)
