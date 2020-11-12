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
from typing import Any, Dict, Optional, List

from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_in
from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema
from .store import DataStore
from .store import new_data_store

# def _store_config_factory(**kwargs):
#     return DataStoreConfig.from_dict(kwargs)
#
#
# def _store_config_serializer(store_config: 'DataStoreConfig'):
#     return store_config.to_dict()


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
    # factory=_store_config_factory,
    # serializer=_store_config_serializer,
)

# def _store_pool_factory(**kwargs):
#     return DataStorePool.from_dict(kwargs)
#
#
# def _store_pool_serializer(store_pool: 'DataStorePool'):
#     return store_pool.to_dict()


DATA_STORE_POOL_SCHEMA = JsonObjectSchema(
    additional_properties=DATA_STORE_CONFIG_SCHEMA,
    # factory=_store_pool_factory,
    # serializer=_store_pool_serializer,
)


class DataStoreConfig:

    def __init__(self,
                 store_id: str,
                 store_params: Dict[str, Any] = None,
                 name: str = None,
                 description: str = None):
        assert_given(store_id, name='store_id')
        if store_params is not None:
            assert_instance(store_params, dict, name='store_params')
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
                               store_params=d.get('store_params'),
                               name=d.get('name'),
                               description=d.get('description'))

    def to_dict(self) -> Dict[str, Any]:
        d = dict(store_id=self._store_id)
        if self._store_params:
            d.update(store_params=self._store_params)
        if self._name:
            d.update(name=self._name)
        if self._description:
            d.update(description=self._description)
        return d


class _DataStoreInstance:

    def __init__(self, store_config: DataStoreConfig):
        assert_given(store_config, name='store_config')
        assert_instance(store_config, DataStoreConfig, name='store_config')
        self._store_config = store_config
        self._store: Optional[DataStore] = None

    @property
    def store_config(self) -> DataStoreConfig:
        return self._store_config

    @property
    def store(self) -> DataStore:
        if self._store is None:
            self._store = new_data_store(self._store_config.store_id,
                                         **(self._store_config.store_params or {}))
        return self._store

    def close(self):
        store = self._store
        if store is not None and hasattr(store, 'close') and callable(store.close):
            store.close()


class DataStorePool:
    """
    A pool of configured data store instances.

    Actual data store instantiation only takes place lazily.
    A pool is may be created using it :meth:from_dict() (or :meth:from_file())
    which receives a (JSON) dictionary that maps store instance names to
    store configurations:

        {
            "<store_instance_id>": {
                "store_id": "<store_id>",
                "store_params": {
                    "<param_name>": <param_value>,
                    ...
                },
                "name": "<optional_human_readable_name>",
                "description": "<optional_human_readable_description>",
            },
            ...
        }

    :param store_configs: A dictionary that maps store instance identifiers to to store configurations.
    """

    def __init__(self, store_configs: Dict[str, DataStoreConfig] = None):
        if store_configs is not None:
            assert_instance(store_configs, dict, name='stores_configs')
            self._instances: Dict[str, _DataStoreInstance] = {k: _DataStoreInstance(v) for k, v in store_configs.items()}
        else:
            self._instances: Dict[str, _DataStoreInstance] = {}

    @property
    def store_instance_ids(self) -> List[str]:
        return sorted([k for k, v in self._instances.items()])

    @property
    def store_configs(self) -> List[DataStoreConfig]:
        return [v.store_config for k, v in self._instances.items()]

    def has_store_config(self, store_instance_id: str) -> bool:
        assert_given(store_instance_id, 'store_instance_id')
        return store_instance_id in self._instances

    def add_store_config(self, store_instance_id: str, store_config: DataStoreConfig):
        assert_given(store_instance_id, 'store_instance_id')
        assert_given(store_config, 'store_config')
        if store_instance_id in self._instances:
            self._instances[store_instance_id].close()
        self._instances[store_instance_id] = _DataStoreInstance(store_config)

    def remove_store_config(self, store_instance_id: str):
        assert_given(store_instance_id, 'store_instance_id')
        assert_in(store_instance_id, self._instances, 'store_instance_id')
        self._instances[store_instance_id].close()
        del self._instances[store_instance_id]

    def get_store_config(self, store_instance_id: str) -> DataStoreConfig:
        return self._instances[store_instance_id].store_config

    def get_store(self, store_instance_id: str) -> DataStore:
        return self._instances[store_instance_id].store

    def close_all_stores(self):
        for instance in self._instances.values():
            instance.close()

    @classmethod
    def from_file(cls, path: str) -> 'DataStorePool':
        with open(path) as fp:
            store_configs = json.load(fp)
        return cls.from_dict(store_configs)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DataStorePool':
        DATA_STORE_POOL_SCHEMA.validate_instance(d)
        return cls({k: DataStoreConfig.from_dict(v) for k, v in d.items()})

    def to_dict(self) -> Dict[str, Any]:
        return {instance_id: instance.store_config.to_dict() for instance_id, instance in self._instances.items()}
