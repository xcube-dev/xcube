import unittest

import jsonschema

from xcube.core.store import DataStoreConfig
from xcube.core.store import DataStorePool


class DataStoreConfigTest(unittest.TestCase):
    def test_constr_and_props(self):
        store_config = DataStoreConfig('directory', store_params={'base_dir': '.'}, name='Local',
                                       description='Local files')
        self.assertEqual('directory', store_config.store_id)
        self.assertEqual({'base_dir': '.'}, store_config.store_params)
        self.assertEqual('Local', store_config.name)
        self.assertEqual('Local files', store_config.description)


class DataStorePoolTest(unittest.TestCase):

    def test_empty_config(self):
        store_configs = {
        }
        pool = DataStorePool.from_dict(store_configs)
        self.assertIsInstance(pool, DataStorePool)
        self.assertEqual([], pool.store_instance_ids)
        self.assertEqual([], pool.store_configs)

    def test_no_store_params(self):
        store_configs = {
            "mem_1": {
                "store_id": "memory"
            }
        }
        pool = DataStorePool.from_dict(store_configs)
        self.assertIsInstance(pool, DataStorePool)
        self.assertEqual(["mem_1"], pool.store_instance_ids)

    def test_no_store_id(self):
        store_configs = {
            "mem_1": {
            }
        }
        with self.assertRaises(jsonschema.exceptions.ValidationError) as cm:
            DataStorePool.from_dict(store_configs)
        self.assertEqual(0, f'{cm.exception}'.index("'store_id' is a required property\n"))

    def test_multi_stores_with_params(self):
        store_configs = {
            "test-store": {
                "store_id": "memory",
            },
            "local-datacubes-1": {
                "store_id": "directory",
                "store_params": {
                    "base_dir": "/home/bibo/datacubes-1",
                }
            },
            "local-datacubes-2": {
                "store_id": "directory",
                "store_params": {
                    "base_dir": "/home/bibo/datacubes-2",
                }
            },
        }
        pool = DataStorePool.from_dict(store_configs)
        self.assertIsInstance(pool, DataStorePool)
        self.assertEqual(["local-datacubes-1", "local-datacubes-2", "test-store"], pool.store_instance_ids)
        for instance_id in pool.store_instance_ids:
            self.assertTrue(pool.has_store_config(instance_id))
            self.assertIsInstance(pool.get_store_config(instance_id), DataStoreConfig)
