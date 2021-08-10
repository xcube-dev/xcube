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
import os.path
from typing import Any, Dict, Optional, List, Union

import yaml

from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonIntegerSchema
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema
from .store import DataStore
from .store import DataStoreError
from .store import new_data_store


def get_data_store_instance(store_id: str,
                            store_params: Dict[str, Any] = None,
                            store_pool: 'DataStorePool' = None) -> 'DataStoreInstance':
    """
    Get a data store instance for identifier *store_id*.

    If *store_id* is prefixed by a "@", it is an "instance identifier".
    In this case the store instance is retrieved from the expected *store_pool* argument.
    Otherwise a new store instance is created using optional *store_params*.

    :param store_id: Store identifier, may be prefixed by a "@" to indicate a store instance identifier.
    :param store_params: Store parameters, only valid if *store_id* is not an instance identifier.
    :param store_pool: A pool of configured store instances used if *store_id* is an instance identifier.
    :return: a DataStoreInstance object
    :raise: DataStoreError if a configured store does not exist
    """
    if store_id.startswith('@'):
        store_instance_id = store_id[1:]
        if store_pool is None:
            raise ValueError(f'store_pool must be given,'
                             f' with store_id ("{store_id}") referring to a configured store')
        if store_params:
            raise ValueError(f'store_params cannot be given,'
                             f' with store_id ("{store_id}") referring to a configured store')
        return store_pool.get_store_instance(store_instance_id)
    return DataStoreInstance(DataStoreConfig(store_id, store_params))


DATA_STORE_CONFIG_SCHEMA = JsonObjectSchema(
    properties=dict(
        store_id=JsonStringSchema(min_length=1),
        store_params=JsonObjectSchema(
            additional_properties=True
        ),
        title=JsonStringSchema(min_length=1),
        description=JsonStringSchema(min_length=1),
        cost_params=JsonObjectSchema(
            properties=dict(
                input_pixels_per_punit=JsonIntegerSchema(minimum=1),
                output_pixels_per_punit=JsonIntegerSchema(minimum=1),
                input_punits_weight=JsonNumberSchema(exclusive_minimum=0.0, default=1.0),
                output_punits_weight=JsonNumberSchema(exclusive_minimum=0.0, default=1.0),
            ),
            additional_properties=False,
            required=['input_pixels_per_punit', 'output_pixels_per_punit'],
        )
    ),
    required=['store_id'],
)

DATA_STORE_POOL_SCHEMA = JsonObjectSchema(
    additional_properties=DATA_STORE_CONFIG_SCHEMA,
)


class DataStoreConfig:
    """
    The configuration of a data store.
    The class is used by :class:DataStorePool to instantiate stores in a deferred manner.

    :param store_id: the data store identifier
    :param store_params: optional store parameters
    :param title: a human-readable title for the store instance
    :param description: a human-readable description of the store instance
    """

    def __init__(self,
                 store_id: str,
                 store_params: Dict[str, Any] = None,
                 title: str = None,
                 description: str = None):
        assert_given(store_id, name='store_id')
        if store_params is not None:
            assert_instance(store_params, dict, name='store_params')
        self._store_id = store_id
        self._store_params = store_params
        self._title = title
        self._description = description

    @property
    def store_id(self) -> Optional[str]:
        return self._store_id

    @property
    def store_params(self) -> Optional[Dict[str, Any]]:
        return self._store_params

    @property
    def title(self) -> Optional[str]:
        return self._title

    @property
    def description(self) -> Optional[str]:
        return self._description

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DataStoreConfig':
        DATA_STORE_CONFIG_SCHEMA.validate_instance(d)
        return DataStoreConfig(d['store_id'],
                               store_params=d.get('store_params'),
                               title=d.get('title'),
                               description=d.get('description'))

    def to_dict(self) -> Dict[str, Any]:
        d = dict(store_id=self._store_id)
        if self._store_params:
            d.update(store_params=self._store_params)
        if self._title:
            d.update(name=self._title)
        if self._description:
            d.update(description=self._description)
        return d


class DataStoreInstance:
    """
    Internal class used by DataStorePool to maintain store configurations + instances.
    """

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


DataStorePoolLike = Union[str, Dict, 'DataStorePool']


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
                "title": "<optional_human_readable_title>",
                "description": "<optional_human_readable_description>",
            },
            ...
        }

    :param store_configs: A dictionary that maps store instance identifiers to to store configurations.
    """

    def __init__(self, store_configs: Dict[str, DataStoreConfig] = None):
        if store_configs is not None:
            assert_instance(store_configs, dict, name='stores_configs')
            self._instances: Dict[str, DataStoreInstance] = {k: DataStoreInstance(v) for k, v in
                                                             store_configs.items()}
        else:
            self._instances: Dict[str, DataStoreInstance] = {}

    @property
    def store_instance_ids(self) -> List[str]:
        return sorted([k for k, v in self._instances.items()])

    @property
    def store_configs(self) -> List[DataStoreConfig]:
        return [v.store_config for k, v in self._instances.items()]

    def has_store_config(self, store_instance_id: str) -> bool:
        assert_instance(store_instance_id, str, 'store_instance_id')
        return store_instance_id in self._instances

    def add_store_config(self, store_instance_id: str, store_config: DataStoreConfig):
        assert_instance(store_instance_id, str, 'store_instance_id')
        assert_instance(store_config, DataStoreConfig, 'store_config')
        if store_instance_id in self._instances:
            self._instances[store_instance_id].close()
        self._instances[store_instance_id] = DataStoreInstance(store_config)

    def remove_store_config(self, store_instance_id: str):
        self._assert_valid_instance_id(store_instance_id)
        self._instances[store_instance_id].close()
        del self._instances[store_instance_id]

    def get_store_config(self, store_instance_id: str) -> DataStoreConfig:
        self._assert_valid_instance_id(store_instance_id)
        return self._instances[store_instance_id].store_config

    def get_store(self, store_instance_id: str) -> DataStore:
        self._assert_valid_instance_id(store_instance_id)
        return self._instances[store_instance_id].store

    def get_store_instance(self, store_instance_id: str) -> DataStoreInstance:
        self._assert_valid_instance_id(store_instance_id)
        return self._instances[store_instance_id]

    def close_all_stores(self):
        for instance in self._instances.values():
            instance.close()

    @classmethod
    def normalize(cls, data_store_pool: DataStorePoolLike) \
            -> 'DataStorePool':
        """
        Normalize given *data_store_pool* to an instance of
        :class:DataStorePool.

        If *data_store_pool* is already a DataStorePool it is returned as is.
        If it is a ``str``, it is interpreted as a YAML or JSON file path
        and the request is read from file using ``DataStorePool.from_file()``.
        If it is a ``dict``, it is interpreted as a JSON object and the
        request is parsed using ``DataStorePool.from_dict()``.

        :param data_store_pool The data store pool instance,
            or data stores configuration file path, or data store pool
            JSON object.
        :raise TypeError if *data_store_pool* is not a ``CubeGeneratorRequest``,
            ``str``, or ``dict``.
        """
        if isinstance(data_store_pool, DataStorePool):
            return data_store_pool
        if isinstance(data_store_pool, str):
            return DataStorePool.from_file(data_store_pool)
        if isinstance(data_store_pool, dict):
            return DataStorePool.from_dict(data_store_pool)
        raise TypeError('data_store_pool must be a str, dict, '
                        'or a DataStorePool instance')

    @classmethod
    def from_file(cls, path: str) -> 'DataStorePool':
        _, ext = os.path.splitext(path)
        with open(path) as fp:
            if ext == '.json':
                store_configs = json.load(fp)
            else:
                store_configs = yaml.safe_load(fp)
        return cls.from_dict(store_configs or {})

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DataStorePool':
        DATA_STORE_POOL_SCHEMA.validate_instance(d)
        return cls({k: DataStoreConfig.from_dict(v) for k, v in d.items()})

    def to_dict(self) -> Dict[str, Any]:
        return {instance_id: instance.store_config.to_dict() for instance_id, instance in self._instances.items()}

    def _assert_valid_instance_id(self, store_instance_id: str):
        assert_instance(store_instance_id, str, name='store_instance_id')
        if store_instance_id not in self._instances:
            raise DataStoreError(f'Configured data store instance "{store_instance_id}" not found.')
