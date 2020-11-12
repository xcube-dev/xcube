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
from typing import Any, Dict, Optional

from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema
from .store import DataStore, DataStoreError
from .store import new_data_store
from ...util.assertions import assert_given


def _store_factory(store_id: str = None,
                   store_params: Dict[str, Any] = None):
    return new_data_store(store_id, **(store_params or {}))


DATA_STORE_CONFIG_SCHEMA = JsonObjectSchema(
    properties=dict(
        store_id=JsonStringSchema(min_length=1),
        store_params=JsonObjectSchema(
            additional_properties=True
        ),
        name=JsonStringSchema(min_length=1),
        description=JsonStringSchema(min_length=1),
    ),
    required=['store_id'],
    factory=_store_factory,
)

_STORE_CONFIG_SCHEMA = JsonObjectSchema(
    additional_properties=DATA_STORE_CONFIG_SCHEMA,
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
            "<store_instance_id>": {
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


class DataStoreConfig:

    def __init__(self,
                 store_id: str,
                 store_params: Dict[str, Any] = None,
                 name: str = None,
                 description: str = None):
        assert_given(store_id, 'store_id')
        self._store_id = store_id
        self._store_params = store_params
        self._name = name
        self._description = description

    @property
    def store_id(self) -> Optional[str]:
        return self._store_id

    @property
    def store_params(self) -> Optional[Dict[str, Any]]:
        return self._store_params

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def description(self) -> Optional[str]:
        return self._description

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DataStoreConfig':
        DATA_STORE_CONFIG_SCHEMA.validate_instance(d)
        return DataStoreConfig(d['store_id'],
                               d.get('store_params'),
                               d.get('name'),
                               d.get('description'))

    def to_dict(self) -> Dict[str, Any]:
        d = dict(store_id=self._store_id)
        if self._store_params:
            d.update(store_params=self._store_params)
        if self._name:
            d.update(name=self._name)
        if self._description:
            d.update(description=self._description)
        return d


class DataStoreInstance:

    def __init__(self, config: DataStoreConfig):
        self._config = config
        self._store = None

    @property
    def config(self) -> DataStoreConfig:
        return self._config

    @property
    def store(self) -> DataStoreConfig:
        if self._store is None:
            self._store = _STORE_CONFIG_SCHEMA.from_instance(self._config.to_dict())
        return self._store


class DataStoreInstance:

    def __init__(self, store_config:):
        self._store
