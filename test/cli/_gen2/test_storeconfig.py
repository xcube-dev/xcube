import unittest

import jsonschema

from xcube.cli._gen2.storeconfig import new_data_store_instances
from xcube.core.store import DataStore


class StoreConfTest(unittest.TestCase):

    def test_empty_config(self):
        store_configs = {
        }
        store_instances = new_data_store_instances(store_configs)
        self.assertEqual({}, store_instances)

    def test_no_store_params(self):
        store_configs = {
            "mem_1": {
                "store_id": "memory"
            }
        }
        store_instances = new_data_store_instances(store_configs)
        self.assertIsInstance(store_instances, dict)
        self.assertIn('mem_1', store_instances)
        self.assertIsInstance(store_instances['mem_1'], DataStore)

    def test_no_store_id(self):
        store_configs = {
            "mem_1": {
            }
        }
        with self.assertRaises(jsonschema.exceptions.ValidationError) as cm:
            new_data_store_instances(store_configs)
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
        store_instances = new_data_store_instances(store_configs)
        self.assertIsInstance(store_instances, dict)
        for name in ["test-store", "local-datacubes-1", "local-datacubes-2"]:
            self.assertIn(name, store_instances)
            self.assertIsInstance(store_instances[name], DataStore)
