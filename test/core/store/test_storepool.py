import json
import unittest

import jsonschema

from xcube.core.store import DataStore
from xcube.core.store import DataStoreConfig
from xcube.core.store import DataStoreError
from xcube.core.store import DataStorePool
from xcube.core.store import get_data_store
from xcube.core.store.stores.directory import DirectoryDataStore


class GetDataStoreTest(unittest.TestCase):

    def test_get_data_store_new_inst(self):
        store = get_data_store('directory', store_params=dict(base_dir='.'))
        self.assertIsInstance(store, DirectoryDataStore)
        store2 = get_data_store('directory', store_params=dict(base_dir='.'))
        self.assertIsNot(store, store2)

    def test_get_data_store_from_pool(self):
        pool = DataStorePool({'dir': DataStoreConfig('directory', store_params=dict(base_dir='.'))})
        store = get_data_store('@dir', store_pool=pool)
        self.assertIsInstance(store, DirectoryDataStore)
        store2 = get_data_store('@dir', store_pool=pool)
        self.assertIs(store, store2)

    def test_get_data_store_from_pool_with_params(self):
        pool = DataStorePool({'@dir': DataStoreConfig('directory', store_params=dict(base_dir='.'))})
        with self.assertRaises(ValueError) as cm:
            get_data_store('@dir', store_pool=pool, store_params={'thres': 5})
        self.assertEqual('store_params cannot be given, with store_id ("@dir") referring to a configured store',
                         f'{cm.exception}')

    def test_get_data_store_from_pool_without_pool(self):
        with self.assertRaises(ValueError) as cm:
            get_data_store('@dir')
        self.assertEqual('store_pool must be given, with store_id ("@dir") referring to a configured store',
                         f'{cm.exception}')


class DataStoreConfigTest(unittest.TestCase):

    def test_constructor_and_instance_props(self):
        store_config = DataStoreConfig('directory', store_params={'base_dir': '.'}, title='Local',
                                       description='Local files')
        self.assertEqual('directory', store_config.store_id)
        self.assertEqual({'base_dir': '.'}, store_config.store_params)
        self.assertEqual('Local', store_config.title)
        self.assertEqual('Local files', store_config.description)

    def test_constructor_asserts(self):
        with self.assertRaises(ValueError) as cm:
            DataStoreConfig('')
        self.assertEqual('store_id must be given', f'{cm.exception}')

        with self.assertRaises(TypeError) as cm:
            # noinspection PyTypeChecker
            DataStoreConfig('directory', store_params=[1, 'B'])
        self.assertEqual("store_params must be an instance of <class 'dict'>", f'{cm.exception}')

    def test_to_dict(self):
        store_config = DataStoreConfig('directory', store_params={'base_dir': '.'}, title='Local',
                                       description='Local files')
        self.assertEqual({'description': 'Local files',
                          'name': 'Local',
                          'store_id': 'directory',
                          'store_params': {'base_dir': '.'}},
                         store_config.to_dict())

    def test_from_dict(self):
        store_config = DataStoreConfig.from_dict({'description': 'Local files',
                                                  'name': 'Local',
                                                  'store_id': 'directory',
                                                  'store_params': {'base_dir': '.'}})
        self.assertIsInstance(store_config, DataStoreConfig)
        self.assertEqual('directory', store_config.store_id)
        self.assertEqual({'base_dir': '.'}, store_config.store_params)
        self.assertEqual('Local', store_config.title)
        self.assertEqual('Local files', store_config.description)


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
        self.assertTrue("'store_id' is a required property" in f'{cm.exception}', msg=f'{cm.exception}')

        store_configs = {
            "dir": {
                "store_id": 10
            }
        }
        with self.assertRaises(jsonschema.exceptions.ValidationError) as cm:
            DataStorePool.from_dict(store_configs)
        self.assertTrue("Failed validating 'type' in schema" in f'{cm.exception}', msg=f'{cm.exception}')

    def test_from_file(self):
        store_configs = {
            "ram-1": {
                "store_id": "memory"
            },
            "ram-2": {
                "store_id": "memory"
            }
        }
        path = 'test-store-configs.json'
        with open(path, 'w') as fp:
            json.dump(store_configs, fp, indent=2)
        try:
            pool = DataStorePool.from_file(path)
            self.assertIsInstance(pool, DataStorePool)
            self.assertEqual(['ram-1', 'ram-2'], pool.store_instance_ids)
        finally:
            import os
            os.remove(path)

    def test_get_store(self):
        store_configs = {
            "dir-1": {
                "store_id": "directory",
                "store_params": {
                    "base_dir": "bibo"
                }
            },
        }
        pool = DataStorePool.from_dict(store_configs)
        store = pool.get_store('dir-1')
        self.assertIsInstance(store, DirectoryDataStore)
        self.assertEqual('bibo', store.base_dir)
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
                          'dir': {'store_id': 'directory',
                                  'store_params': {'base_dir': 'bibo'}}},
                         DataStorePool({'ram': DataStoreConfig(store_id='memory'),
                                        'dir': DataStoreConfig(store_id='directory',
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
                "store_id": "directory",
                "store_params": {
                    "base_dir": "/home/bibo/datacubes-1",
                }
            },
            "local-2": {
                "store_id": "directory",
                "store_params": {
                    "base_dir": "/home/bibo/datacubes-2",
                }
            },
        }
        pool = DataStorePool.from_dict(store_configs)
        self.assertIsInstance(pool, DataStorePool)
        self.assertEqual(["local-1", "local-2", "ram-1", "ram-2"], pool.store_instance_ids)
        for instance_id in pool.store_instance_ids:
            self.assertTrue(pool.has_store_config(instance_id))
            self.assertIsInstance(pool.get_store_config(instance_id), DataStoreConfig)
            self.assertIsInstance(pool.get_store(instance_id), DataStore)
