# The MIT License (MIT)
# Copyright (c) 2020 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import json
from typing import List, Sequence

import click

from xcube.constants import EXTENSION_POINT_DATA_OPENERS
from xcube.constants import EXTENSION_POINT_DATA_STORES
from xcube.constants import EXTENSION_POINT_DATA_WRITERS
from xcube.util.plugin import get_extension_registry

_TAB = ' \t'


@click.command(name='list')
def store_list():
    """List names of data stores."""
    print(f'Data stores:')
    count = _dump_data_stores()
    print(f'{count} data store{"s" if count != 1 else ""} found.')


@click.command(name='list')
def opener_list():
    """List names of data openers."""
    print(f'Data openers:')
    count = _dump_data_openers()
    print(f'{count} data opener{"s" if count != 1 else ""} found.')


@click.command(name='list')
def writer_list():
    """List names of data writers."""
    print(f'Data writers:')
    count = _dump_data_writers()
    print(f'{count} data writer{"s" if count != 1 else ""} found.')


@click.command(name='info')
@click.argument('store_id', metavar='STORE')
@click.argument('store_params', metavar='PARAMS', nargs=-1)
@click.option('-P', '--params', 'show_params', is_flag=True, help='Show data store parameters.')
@click.option('-O', '--openers', 'show_openers', is_flag=True, help='Show available data store openers.')
@click.option('-W', '--writers', 'show_writers', is_flag=True, help='Show available data store writers.')
@click.option('-D', '--data', 'show_data_ids', is_flag=True, help='Show available data resources.')
@click.option('-j', '--json', 'use_json_format', is_flag=True, help='Output using JSON format.')
def store_info(store_id: str,
               store_params: List[str],
               show_params: bool,
               show_openers: bool,
               show_writers: bool,
               show_data_ids: bool,
               use_json_format: bool):
    """
    Show data store information.

    Dumps detailed data store information in human readable form or as JSON, when using the --json option.

    You can obtain valid STORE names using command "xcube store list".

    Note some stores require provision of parameters PARAMS
    when using one of the options --openers, --writers, or --data.
    To find out which parameters are available use the command with just the --params option first.
    """
    extension = get_extension_registry().get_extension(EXTENSION_POINT_DATA_STORES, store_id)
    from xcube.core.store import get_data_store_params_schema
    from xcube.core.store import MutableDataStore
    params_schema = get_data_store_params_schema(store_id)
    description = extension.metadata.get('description')
    requires_store_instance = any((show_openers, show_writers, show_data_ids))
    data_store = _new_data_store(store_id, store_params) if requires_store_instance else None
    if use_json_format:
        d = dict()
        d['store_id'] = store_id
        if description:
            d['description'] = description
        if show_params:
            d['params_schema'] = params_schema.to_dict()
        if show_openers:
            d['opener_ids'] = data_store.get_data_opener_ids()
        if show_writers and isinstance(data_store, MutableDataStore):
            d['writer_ids'] = data_store.get_data_writer_ids()
        if show_data_ids:
            d['data_ids'] = list(data_store.get_data_ids())
        if show_openers:
            print(json.dumps(d, indent=2))
    else:
        if description:
            print(description)
        if show_params:
            print(_format_params_schema(params_schema))
        if show_openers:
            print(f'Data openers:')
            _dump_store_openers(data_store)
        if show_writers:
            if isinstance(data_store, MutableDataStore):
                print(f'Data writers:')
                _dump_store_writers(data_store)
            else:
                print(f'No writers available, because data store "{store_id}" is not mutable.')
        if show_data_ids:
            print(f'Data resources:')
            count = _dump_store_data_ids(data_store)
            print(f'{count} data resource{"s" if count != 1 else ""} found.')


@click.command(name='data')
@click.argument('store_id', metavar='STORE')
@click.argument('data_id', metavar='ID')
@click.argument('store_params', metavar='PARAMS', nargs=-1)
def store_data(store_id: str, data_id: str, store_params: List[str]):
    """
    Show data resource information.
    """
    data_store = _new_data_store(store_id, store_params)
    data_descriptor = data_store.describe_data(data_id)
    print(f'Descriptor for data resource "{data_id}" in data store "{store_id}":')
    print(json.dumps(data_descriptor.to_dict(), indent=2))


@click.command(name='info')
@click.argument('opener_id', metavar='OPENER')
def opener_info(opener_id: str):
    """
    Show data opener information.
    You can obtain valid OPENER names using command "xcube opener list".
    """
    extension = get_extension_registry().get_extension(EXTENSION_POINT_DATA_WRITERS, opener_id)
    description = extension.metadata.get('description')
    if description:
        print(description)
    from xcube.core.store import new_data_opener
    writer = new_data_opener(opener_id)
    params_schema = writer.get_open_data_params_schema()
    print(_format_params_schema(params_schema))


@click.command(name='info')
@click.argument('writer_id', metavar='WRITER')
def writer_info(writer_id: str):
    """
    Show data opener information.
    You can obtain valid WRITER names using command "xcube writer list".
    """
    extension = get_extension_registry().get_extension(EXTENSION_POINT_DATA_WRITERS, writer_id)
    description = extension.metadata.get('description')
    if description:
        print(description)
    from xcube.core.store import new_data_writer
    writer = new_data_writer(writer_id)
    params_schema = writer.get_write_data_params_schema()
    print(_format_params_schema(params_schema))


@click.group()
def store():
    """
    Tools for data stores.
    """
    pass


@click.group()
def opener():
    """
    Tools for data openers.
    """
    pass


@click.group()
def writer():
    """
    Tools for data writers.
    """
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
    """
    Tools for xcube's I/O system.
    """
    pass


io.add_command(store)
io.add_command(opener)
io.add_command(writer)


# from xcube.util.jsonschema import JsonObjectSchema


def _format_params_schema(params_schema: 'xcube.util.jsonschema.JsonObjectSchema') -> str:
    text = []
    if params_schema.properties:
        text.append('Parameters:')
        for param_name, param_schema in params_schema.properties.items():
            text.append(f'{"* " if param_name in params_schema.required else "  "}'
                        f'{param_name}{_TAB}{_format_param_schema(param_schema)}')
    else:
        text.append(f'No parameters required.')
    return '\n'.join(text)


def _format_required_params_schema(params_schema: 'xcube.util.jsonschema.JsonObjectSchema') -> str:
    text = ['Required parameters:']
    for param_name, param_schema in params_schema.properties.items():
        if param_name in params_schema.required:
            text.append(f'  {param_name}{_TAB}{_format_param_schema(param_schema)}')
    return '\n'.join(text)


def _format_param_schema(param_schema: 'xcube.util.jsonschema.JsonSchema'):
    from xcube.util.undefined import UNDEFINED
    param_info = []
    if param_schema.title or param_schema.description:
        description = param_schema.title or param_schema.description
        param_info.append(description + ('' if description.endswith('.') else '.'))
    if param_schema.enum:
        param_info.append(f'Must be one of {", ".join(map(json.dumps, param_schema.enum))}.')
    if param_schema.const is not UNDEFINED:
        param_info.append(f'Must be {json.dumps(param_schema.const)}.')
    if param_schema.default is not UNDEFINED:
        param_info.append(f'Defaults to {json.dumps(param_schema.default)}.')
    param_info_text = ' ' + " ".join(param_info) if param_info else ''
    return f'({param_schema.type}){param_info_text}'


def _dump_data_stores() -> int:
    return _dump_extensions(EXTENSION_POINT_DATA_STORES)


def _dump_data_openers() -> int:
    return _dump_extensions(EXTENSION_POINT_DATA_OPENERS)


def _dump_data_writers() -> int:
    return _dump_extensions(EXTENSION_POINT_DATA_WRITERS)


def _dump_extensions(point: str) -> int:
    count = 0
    for extension in get_extension_registry().find_extensions(point):
        print(f'  {extension.name}{_TAB}{extension.metadata.get("description", "<no description>")}')
        count += 1
    return count


def _dump_store_openers(data_store: 'xcube.core.store.DataStore', data_id: str = None) -> int:
    return _dump_named_extensions(EXTENSION_POINT_DATA_OPENERS, data_store.get_data_opener_ids(data_id=data_id))


def _dump_store_writers(data_store: 'xcube.core.store.DataStore') -> int:
    return _dump_named_extensions(EXTENSION_POINT_DATA_WRITERS, data_store.get_data_writer_ids())


def _dump_store_data_ids(data_store: 'xcube.core.store.DataStore') -> int:
    count = 0
    for data_id in data_store.get_data_ids():
        print(f'  {data_id}')
        count += 1
    return count


def _dump_named_extensions(point: str, names: Sequence[str]) -> int:
    count = 0
    for name in names:
        extension = get_extension_registry().get_extension(point, name)
        if extension:
            print(f'  {name}{_TAB}{extension.metadata.get("description", "<no description>")}')
        else:
            print(f'  {name}{_TAB}<unknown!>')
        count += 1
    return count


def _dump_data_resources(data_store: 'xcube.core.store.DataStore') -> int:
    count = 0
    for data_id in data_store.get_data_ids():
        print(f'  {data_id}')
        count += 1
    return count


def _new_data_store(store_id: str, store_params: List[str]):
    from xcube.core.store import get_data_store_params_schema
    from xcube.core.store import new_data_store
    params_schema = get_data_store_params_schema(store_id)
    store_params_dict = dict()
    if store_params:
        for p_assignment in store_params:
            p_name_value = p_assignment.split('=', maxsplit=2)
            p_name = p_name_value[0].strip()
            if not p_name:
                raise click.ClickException(f'Invalid parameter assignment: {p_assignment}')
            if len(p_name_value) == 2:
                # Passed as name=value
                p_value = p_name_value[1].strip()
                try:
                    p_value = json.loads(p_value)
                except json.decoder.JSONDecodeError:
                    p_value = f'"{p_value}"'
            else:
                # Passed name as flag
                p_value = True
            store_params_dict[p_name] = p_value
    elif params_schema.required:
        raise click.ClickException(f'Data store "{store_id}" has required parameters, but none were given.\n'
                                   f'{_format_required_params_schema(params_schema)}')
    return new_data_store(store_id, **store_params_dict)
