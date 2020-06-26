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
from typing import Any, Dict

from xcube.core.store import DataStore, DataStoreError
from xcube.core.store import new_data_store
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema


def _store_factory(store_id: str = None,
                   store_params: Dict[str, Any] = None):
    return new_data_store(store_id, **(store_params or {}))


_STORE_INSTANCE_SCHEMA = JsonObjectSchema(
    properties=dict(
        store_id=JsonStringSchema(min_length=1),
        store_params=JsonObjectSchema(
            additional_properties=True
        )
    ),
    required=['store_id'],
    factory=_store_factory,
)

_STORE_CONFIG_SCHEMA = JsonObjectSchema(
    additional_properties=_STORE_INSTANCE_SCHEMA,
)


def load_data_store_instances(store_configs_path: str = None):
    if store_configs_path:
        with open(store_configs_path) as fp:
            store_configs = json.load(fp)
        return new_data_store_instances(store_configs)
    return None


def new_data_store_instances(store_configs: Dict[str, Any]) -> Dict[str, DataStore]:
    """
    Create named data store instances from the given dictionary *store_configs*.
    *store_configs* is expected to be the dictionary representation of a JSON object
    that maps store names to parameterized data store instances:

        {
            "<store_name>": {
                "store_id": "<store_id>",
                "store_params": {
                    "<param_name>": <param_value>,
                    ...
                }
            },
            ...
        }

    :param store_configs: A dictionary that maps store names to store configurations.
    :return: A dictionary that maps store names to instantiated stores.
    """
    return _STORE_CONFIG_SCHEMA.from_instance(store_configs)


def get_data_store_instance(store_id: str,
                            store_params: Dict[str, Any],
                            store_instances: Dict[str, DataStore] = None):
    """
    Get data store with identifier *store_id*. If *store_id* is prefixed by a "@", it is the name of a pre-configured
    data store in *store_instances*. Otherwise a new store instance is created using *store_params*.

    :param store_id: store identifier, may be prefixed by a "@".
    :param store_params: store parameters. only valid if *store_id* is not prefixed by a "@".
    :param store_instances: pre-configured store instances used when *store_id*  is prefixed by a "@".
    :return: a data store instance
    """
    if store_id.startswith('@'):
        store_instance_name = store_id[1:]
        if store_instances and store_instance_name in store_instances:
            if store_params:
                raise DataStoreError(f'store "{store_id}": "store_params" '
                                     f'cannot be given for pre-configured data stores')
            return store_instances[store_instance_name]
        else:
            raise DataStoreError(f'pre-configured data store "{store_instance_name}" not found')
    return new_data_store(store_id, **store_params)
