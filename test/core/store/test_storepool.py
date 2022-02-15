import json
import os
import unittest

import jsonschema
import yaml

from xcube.core.store import DataStore
from xcube.core.store import DataStoreConfig
from xcube.core.store import DataStoreError
from xcube.core.store import DataStoreInstance
from xcube.core.store import DataStorePool
from xcube.core.store import get_data_store_instance


class GetDataStoreTest(unittest.TestCase):

    def test_get_data_store_instance_new_inst(self):
        instance = get_data_store_instance('memory')
        self.assertIsInstance(instance, DataStoreInstance)
        self.assertIsInstance(instance.store, DataStore)
        instance2 = get_data_store_instance('memory')
        self.assertIsNot(instance, instance2)
        self.assertIsNot(instance.store, instance2.store)

    def test_get_data_store_instance_from_pool(self):
        pool = DataStorePool({
            'dir': DataStoreConfig('file',
                                   store_params=dict(root='.'))
        })
        instance = get_data_store_instance('@dir', store_pool=pool)
        self.assertTrue(hasattr(instance.store, 'root'))
        # noinspection PyUnresolvedReferences
        self.assertTrue(os.path.isabs(instance.store.root))
        self.assertTrue(os.path.isdir(instance.store.root))
        instance2 = get_data_store_instance('@dir', store_pool=pool)
        self.assertIs(instance, instance2)

    def test_get_data_store_instance_from_pool_with_params(self):
        pool = DataStorePool({
            '@dir': DataStoreConfig('file',
                                    store_params=dict(root='.'))
        })
        with self.assertRaises(ValueError) as cm:
            get_data_store_instance(
                '@dir', store_pool=pool, store_params={'auto_mkdir': True}
            )
        self.assertEqual('store_params cannot be given,'
                         ' with store_id ("@dir") referring'
                         ' to a configured store',
                         f'{cm.exception}')

    def test_get_data_store_instance_from_pool_without_pool(self):
        with self.assertRaises(ValueError) as cm:
            get_data_store_instance('@dir')
        self.assertEqual('store_pool must be given,'
                         ' with store_id ("@dir") referring'
                         ' to a configured store',
                         f'{cm.exception}')

    def test_normalize(self):
        pool = DataStorePool({
            '@dir': DataStoreConfig('directory',
                                    store_params=dict(root='.'))
        })
        file_path = '_test-data-stores-pool.json'
        with open(file_path, 'w') as fp:
            json.dump(pool.to_dict(), fp)
        try:
            pool_1 = DataStorePool.normalize(file_path)
            self.assertIsInstance(pool_1, DataStorePool)
            pool_2 = DataStorePool.normalize(pool_1)
            self.assertIs(pool_2, pool_1)
            pool_3 = DataStorePool.normalize(pool_2.to_dict())
            self.assertIsInstance(pool_3, DataStorePool)
        finally:
            os.remove(file_path)

        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            DataStorePool.normalize(42)


class DataStoreConfigTest(unittest.TestCase):

    def test_constructor_and_instance_props(self):
        store_config = DataStoreConfig('file',
                                       store_params={'root': '.'},
                                       title='Local',
                                       description='Local files')
        self.assertEqual('file', store_config.store_id)
        self.assertEqual({'root': '.'}, store_config.store_params)
        self.assertEqual('Local', store_config.title)
        self.assertEqual('Local files', store_config.description)

    def test_constructor_asserts(self):
        with self.assertRaises(ValueError) as cm:
            DataStoreConfig('')
        self.assertEqual('store_id must be given', f'{cm.exception}')

        with self.assertRaises(TypeError) as cm:
            # noinspection PyTypeChecker
            DataStoreConfig('directory', store_params=[1, 'B'])
        self.assertEqual("store_params must be an instance"
                         " of <class 'dict'>, was <class 'list'>",
                         f'{cm.exception}')

    def test_to_dict(self):
        store_config = DataStoreConfig('directory',
                                       store_params={'base_dir': '.'},
                                       title='Local',
                                       description='Local files')
        self.assertEqual({'description': 'Local files',
                          'name': 'Local',
                          'store_id': 'directory',
                          'store_params': {'base_dir': '.'}},
                         store_config.to_dict())

    def test_from_dict(self):
        store_config = DataStoreConfig.from_dict({
            'description': 'Local files',
            'title': 'Local',
            'store_id': 'file',
            'store_params': {'root': '.'}
        })
        self.assertIsInstance(store_config, DataStoreConfig)
        self.assertEqual('file', store_config.store_id)
        self.assertEqual({'root': '.'}, store_config.store_params)
        self.assertEqual('Local', store_config.title)
        self.assertEqual('Local files', store_config.description)

    def test_from_dict_with_valid_cost_params(self):
        store_config = DataStoreConfig.from_dict({
            'description': 'Local files',
            'title': 'Local',
            'store_id': 'file',
            'store_params': {'root': '.'},
            'cost_params': {
                'input_pixels_per_punit': 500,
                'output_pixels_per_punit': 100,
                'input_punits_weight': 1.1,
            }
        })
        self.assertIsInstance(store_config, DataStoreConfig)
        self.assertEqual('file', store_config.store_id)
        self.assertEqual({'root': '.'}, store_config.store_params)
        self.assertEqual('Local', store_config.title)
        self.assertEqual('Local files', store_config.description)

    def test_from_dict_with_invalid_cost_params(self):
        with self.assertRaises(DataStoreError):
            DataStoreConfig.from_dict({'description': 'Local files',
                                       'title': 'Local',
                                       'store_id': 'file',
                                       'store_params': {'root': '.'},
                                       'cost_params': {
                                           # Required:
                                           # 'input_pixels_per_punit': 10,
                                           # 'output_pixels_per_punit': 20,
                                       }})


class DataStorePoolTest(unittest.TestCase):
    def test_default_constr(self):
        pool = DataStorePool()
        self.assertEqual([], pool.store_instance_ids)
        self.assertEqual([], pool.store_configs)

    def test_from_dict_empty(self):
        pool = DataStorePool.from_dict({})
        self.assertIsInstance(pool, DataStorePool)
        self.assertEqual([], pool.store_instance_ids)
        self.assertEqual([], pool.store_configs)

    def test_from_dict_no_store_params(self):
        store_configs = {
            "ram-1": {
                "store_id": "memory"
            }
        }
        pool = DataStorePool.from_dict(store_configs)
        self.assertIsInstance(pool, DataStorePool)
        self.assertEqual(["ram-1"], pool.store_instance_ids)
        self.assertIsInstance(pool.get_store_config('ram-1'), DataStoreConfig)

    def test_from_dict_with_bad_dicts(self):
        store_configs = {
            "dir": {
            }
        }
        with self.assertRaises(jsonschema.exceptions.ValidationError) as cm:
            DataStorePool.from_dict(store_configs)
        self.assertTrue(
            "'store_id' is a required property" in f'{cm.exception}',
            msg=f'{cm.exception}')

        store_configs = {
            "dir": {
                "store_id": 10
            }
        }
        with self.assertRaises(jsonschema.exceptions.ValidationError) as cm:
            DataStorePool.from_dict(store_configs)
        self.assertTrue(
            "Failed validating 'type' in schema" in f'{cm.exception}',
            msg=f'{cm.exception}')

    def test_from_json_file(self):
        self._assert_file_ok('json')

    def test_from_yaml_file(self):
        self._assert_file_ok('yaml')

    def test_from_json_file_env(self):
        self._assert_file_ok('json', use_env_vars=True)
        self._assert_file_ok('json',
                             root_1='/test_1',
                             root_2='/test_2',
                             use_env_vars=True)

    def test_from_yaml_file_env(self):
        self._assert_file_ok('yaml', use_env_vars=True)
        self._assert_file_ok('yaml',
                             root_1='/test_1',
                             root_2='/test_2',
                             use_env_vars=True)

    def _assert_file_ok(self,
                        format_name: str,
                        root_1="/root1",
                        root_2="/root2",
                        use_env_vars=False):
        if use_env_vars:
            store_configs = self._get_test_config(
                root_1='${_TEST_ROOT_1}',
                root_2='${_TEST_ROOT_2}'
            )
            import os
            os.environ['_TEST_ROOT_1'] = root_1
            os.environ['_TEST_ROOT_2'] = root_2
        else:
            store_configs = self._get_test_config(
                root_1=root_1,
                root_2=root_2
            )
        path = 'test-store-configs.' + format_name
        with open(path, 'w') as fp:
            mod = yaml if format_name == 'yaml' else json
            mod.dump(store_configs, fp, indent=2)
        try:
            pool = DataStorePool.from_file(path)
            self.assertIsInstance(pool, DataStorePool)
            self.assertEqual(['ram-1', 'ram-2'], pool.store_instance_ids)
            config_1 = pool.get_store_config('ram-1')
            self.assertIsInstance(config_1, DataStoreConfig)
            self.assertEqual(
                {'store_id': 'memory',
                 'store_params': {'root': root_1}},
                config_1.to_dict())
            config_2 = pool.get_store_config('ram-2')
            self.assertIsInstance(config_2, DataStoreConfig)
            self.assertEqual(
                {'store_id': 'memory',
                 'store_params': {'root': root_2}},
                config_2.to_dict())
        finally:
            import os
            os.remove(path)

    @staticmethod
    def _get_test_config(root_1: str, root_2: str):
        return {
            "ram-1": {
                "store_id": "memory",
                "store_params": {
                    "root": root_1
                },
            },
            "ram-2": {
                "store_id": "memory",
                "store_params": {
                    "root": root_2
                },
            }
        }

    def test_get_store(self):
        store_configs = {
            "dir-1": {
                "store_id": "file",
                "store_params": {
                    "root": "./bibo"
                }
            },
        }
        pool = DataStorePool.from_dict(store_configs)
        store = pool.get_store('dir-1')
        self.assertTrue(hasattr(store, 'root'))
        # noinspection PyUnresolvedReferences
        self.assertTrue(os.path.isabs(store.root))
        self.assertFalse(os.path.exists(store.root))
        # Should stay same instance
        self.assertIs(store, pool.get_store('dir-1'))
        self.assertIs(store, pool.get_store('dir-1'))

    def test_get_store_error(self):
        pool = DataStorePool()
        with self.assertRaises(DataStoreError) as cm:
            pool.get_store('dir-1')
        self.assertEqual('Configured data store instance "dir-1" not found.', f'{cm.exception}')

    def test_to_dict(self):
        self.assertEqual({}, DataStorePool().to_dict())
        self.assertEqual({'ram': {'store_id': 'memory'},
                          'dir': {'store_id': 'file',
                                  'store_params': {'base_dir': 'bibo'}}},
                         DataStorePool({'ram': DataStoreConfig(store_id='memory'),
                                        'dir': DataStoreConfig(store_id='file',
                                                               store_params=dict(base_dir="bibo"))
                                        }).to_dict())

    def test_close_all_stores(self):
        store_configs = {
            "ram-1": {
                "store_id": "memory",
            },
        }
        pool = DataStorePool.from_dict(store_configs)
        # Smoke test, we do not expect any visible state changes after close_all_stores()
        pool.close_all_stores()

    def test_add_remove_store_config(self):
        pool = DataStorePool()
        self.assertEqual([], pool.store_instance_ids)
        pool.add_store_config('mem-1', DataStoreConfig('memory'))
        self.assertEqual(['mem-1'], pool.store_instance_ids)
        pool.add_store_config('mem-2', DataStoreConfig('memory'))
        self.assertEqual(['mem-1', 'mem-2'], pool.store_instance_ids)
        pool.add_store_config('mem-1', DataStoreConfig('memory'))
        self.assertEqual(['mem-1', 'mem-2'], pool.store_instance_ids)
        pool.remove_store_config('mem-1')
        self.assertEqual(['mem-2'], pool.store_instance_ids)
        pool.remove_store_config('mem-2')
        self.assertEqual([], pool.store_instance_ids)

    def test_multi_stores_with_params(self):
        """Just test many stores at once"""
        store_configs = {
            "ram-1": {
                "store_id": "memory",
            },
            "ram-2": {
                "store_id": "memory",
            },
            "local-1": {
                "store_id": "file",
                "store_params": {
                    "root": "/home/bibo/datacubes-1",
                }
            },
            "local-2": {
                "store_id": "file",
                "store_params": {
                    "root": "/home/bibo/datacubes-2",
                }
            },
        }
        pool = DataStorePool.from_dict(store_configs)
        self.assertIsInstance(pool, DataStorePool)
        self.assertEqual(["local-1", "local-2", "ram-1", "ram-2"], pool.store_instance_ids)
        for instance_id in pool.store_instance_ids:
            self.assertTrue(pool.has_store_instance(instance_id))
            self.assertIsInstance(pool.get_store_config(instance_id), DataStoreConfig)
            self.assertIsInstance(pool.get_store(instance_id), DataStore)

    def test_get_store_instance_id(self):
        store_params_1 = {
            "root": "./bibo"
        }
        ds_config_1 = DataStoreConfig(store_id='file',
                                      store_params=store_params_1)
        ds_configs = {'dir-1': ds_config_1}
        pool = DataStorePool(ds_configs)

        store_params_2 = {
            "root": "./babo"
        }
        ds_config_2 = DataStoreConfig(store_id='file',
                                      store_params=store_params_2)
        ds_config_3 = DataStoreConfig(store_id='file',
                                      store_params=store_params_1,
                                      title='A third configuration')

        self.assertEqual('dir-1', pool.get_store_instance_id(ds_config_1))
        self.assertEqual('dir-1', pool.get_store_instance_id(ds_config_1,
                                                             strict_check=True))

        self.assertIsNone(pool.get_store_instance_id(ds_config_2))

        self.assertEqual('dir-1', pool.get_store_instance_id(ds_config_3))
        self.assertIsNone(pool.get_store_instance_id(ds_config_3,
                                                     strict_check=True))
