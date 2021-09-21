import os.path
import unittest

from xcube.util.jsonschema import JsonObjectSchema
from xcube.webapi.config import DatasetConfig
from xcube.webapi.config import ServiceConfig


class ServiceConfigTest(unittest.TestCase):
    def test_get_schema(self):
        schema = ServiceConfig.get_schema()
        self.assertIsInstance(schema, JsonObjectSchema)

    def test_from_dict(self):
        test_dir = os.path.dirname(__file__)
        service_config_dict = {
            "Authentication": {
                "Domain": "xcube - dev.eu.auth0.com",
                "Audience": "https://xcube-dev/api/",
            },
            "DatasetAttribution":
                [
                    "© by Brockmann Consult GmbH 2020",
                    "© by EU H2020 CyanoAlert project",
                ],
            "DatasetChunkCacheSize": "100M",
            "Datasets": [
                {
                    # Will only appear for unauthorized clients
                    "Identifier": "local",
                    "Title": "Local OLCI L2C cube for region SNS",
                    "BoundingBox": [0.0, 50, 5.0, 52.5],
                    "FileSystem": "local",
                    "Path": "cube-1-250-250.zarr",
                    "Style": "default",
                    "TimeSeriesDataset": "local_ts",
                    "Augmentation": {
                        "Path": "compute_extra_vars.py",
                        "Function": "compute_variables",
                        "InputParameters": {
                            "factor_chl": 0.2,
                            "factor_tsm": 0.7,
                        },
                    },
                    "PlaceGroups": [
                        {"PlaceGroupRef": "inside-cube"},
                        {"PlaceGroupRef": "outside-cube"},
                    ],
                    "AccessControl": {
                        "IsSubstitute": True,
                    },
                }
            ],
            "DataStores": [
                {
                    "Identifier": "local",
                    "StoreId": "file",
                    "StoreParams": {
                        "root": f"{test_dir}/../../examples/serve/demo",
                    },
                    "Datasets": [
                        {
                            "Path": "*.zarr",
                            "Style": "default"
                        }
                    ]
                },
            ]
        }
        service_config = ServiceConfig.from_dict(service_config_dict)
        self.assertIsInstance(service_config, ServiceConfig)
        self.assertTrue(hasattr(service_config, 'Datasets'))
        self.assertIsInstance(service_config.Datasets, list)
        self.assertEqual(1, len(service_config.Datasets))
        self.assertIsInstance(service_config.Datasets[0], DatasetConfig)
        self.assertTrue(hasattr(service_config.Datasets[0], 'Identifier'))
        self.assertEqual("local", service_config.Datasets[0].Identifier)
