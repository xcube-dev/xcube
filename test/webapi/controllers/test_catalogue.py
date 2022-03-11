import os.path
import unittest
from typing import Any, Optional

from test.webapi.helpers import new_test_service_context
from xcube.webapi.context import ServiceContext
from xcube.webapi.controllers.catalogue import get_color_bars
from xcube.webapi.controllers.catalogue import get_dataset
from xcube.webapi.controllers.catalogue import get_datasets
from xcube.webapi.errors import ServiceBadRequestError, ServiceAuthError


def new_ctx():
    data_store_root = os.path.join(
        os.path.dirname(__file__),
        '..', '..', '..', 'examples', 'serve', 'demo'
    )
    config = dict(
        Authentication=dict(
            Domain='xcube-dev.eu.auth0.com',
            Audience='https://xcube-dev/api/',
        ),
        AccessControl=dict(
            RequiredScopes=[
                'read:dataset:{Dataset}',
                'read:variable:{Variable}'
            ]
        ),
        DataStores=[
            dict(
                Identifier='local',
                StoreId='file',
                StoreParams=dict(
                    root=data_store_root
                ),
                Datasets=[
                    # "base" will only appear for unauthorized clients
                    dict(
                        Identifier='base',
                        Path='cube-1-250-250.zarr',
                        TimeSeriesDataset='local~cube-5-100-200.zarr',
                        AccessControl=dict(
                            IsSubstitute=True
                        )
                    ),
                    # "time_series" will not appear at all,
                    # because it is a "hidden" resource
                    dict(
                        Identifier='time_chunked',
                        Path='cube-5-100-200.zarr',
                        Hidden=True,
                    ),
                ]
            ),
            dict(
                Identifier='remote',
                StoreId='s3',
                StoreParams=dict(
                    root='xcube-examples',
                    storage_options=dict(anon=True),
                ),
                Datasets=[
                    dict(
                        Identifier='base',
                        Path='OLCI-SNS-RAW-CUBE-2.zarr'
                    ),
                ]
            )
        ],
        Datasets=[
            # "local" will only appear for unauthorized clients
            dict(
                Identifier='local_base_1w',
                FileSystem='memory',
                Path='examples/serve/demo/resample_in_time.py',
                Function='compute_dataset',
                InputDatasets=['local~cube-1-250-250.zarr'],
                InputParameters=dict(
                    period='1W',
                    incl_stdev=True
                ),
                AccessControl=dict(
                    IsSubstitute=True
                )
            ),
            dict(
                Identifier='remote_base_1w',
                FileSystem='memory',
                Path='examples/serve/demo/resample_in_time.py',
                Function='compute_dataset',
                InputDatasets=['remote~OLCI-SNS-RAW-CUBE-2.zarr'],
                InputParameters=dict(
                    period='1W',
                    incl_stdev=True
                ),
            ),
        ],
    )
    return ServiceContext(config=config)


class CatalogueControllerTest(unittest.TestCase):
    def test_unauthorized_access(self):
        ctx = new_ctx()
        granted_scopes = None
        response = get_datasets(ctx, granted_scopes=granted_scopes)
        datasets = self._assert_datasets(response)
        datasets_dict = {ds['id']: ds for ds in datasets}
        self.assertEqual(
            {
                'local~cube-1-250-250.zarr',
                'local_base_1w',
                # Not selected, because they require authorisation
                # 'remote~OLCI-SNS-RAW-CUBE-2.zarr',
                # 'remote_base_1w',
            },
            set(datasets_dict)
        )

        dataset = get_dataset(ctx, 'local~cube-1-250-250.zarr')
        self.assertIn('variables', dataset)
        var_dict = {v['name']: v for v in dataset['variables']}
        self.assertEqual({'c2rcc_flags', 'conc_tsm', 'kd489',
                          'conc_chl', 'quality_flags'},
                         set(var_dict.keys()))

        dataset = get_dataset(ctx, 'local_base_1w')
        self.assertIn('variables', dataset)
        var_dict = {v['name']: v for v in dataset['variables']}
        self.assertEqual(set(),
                         set(var_dict.keys()))

        with self.assertRaises(ServiceAuthError) as cm:
            get_dataset(ctx, 'remote_base_1w')
        self.assertEqual(
            'HTTP 401: Missing permission'
            ' (Missing permission'
            ' read:dataset:local_base_1w)',
            f'{cm.exception}'
        )

        with self.assertRaises(ServiceAuthError) as cm:
            get_dataset(ctx, 'remote~OLCI-SNS-RAW-CUBE-2.zarr')
        self.assertEqual(
            'HTTP 401: Missing permission'
            ' (Missing permission'
            ' read:dataset:remote~OLCI-SNS-RAW-CUBE-2.zarr)',
            f'{cm.exception}'
        )

    def test_authorized_access_with_joker_scopes(self):
        ctx = new_ctx()
        granted_scopes = {
            'read:dataset:*',
            'read:variable:*'
        }
        response = get_datasets(ctx, granted_scopes=granted_scopes)
        datasets = self._assert_datasets(response)
        datasets_dict = {ds['id']: ds for ds in datasets}
        self.assertEqual(
            {
                'local_base_1w',
                'local~cube-1-250-250.zarr',
                'remote_base_1w',
                'remote~OLCI-SNS-RAW-CUBE-2.zarr',
            },
            set(datasets_dict)
        )

    def test_authorized_access_with_specific_scopes(self):
        ctx = new_ctx()
        granted_scopes = {
            'read:dataset:remote*',
            'read:variable:*'
        }
        response = get_datasets(ctx, granted_scopes=granted_scopes)
        datasets = self._assert_datasets(response)
        datasets_dict = {ds['id']: ds for ds in datasets}
        self.assertEqual(
            {
                # Not selected, because they are substitutes
                # 'local_base_1w',
                # 'local~cube-1-250-250.zarr',
                'remote_base_1w',
                'remote~OLCI-SNS-RAW-CUBE-2.zarr',
            },
            set(datasets_dict)
        )

    def test_datasets(self):
        ctx = new_test_service_context()
        response = get_datasets(ctx)
        datasets = self._assert_datasets(response, expected_count=2)
        for dataset in datasets:
            self.assertIsInstance(dataset, dict)
            self.assertIn("id", dataset)
            self.assertIn("title", dataset)
            self.assertNotIn("variables", dataset)
            self.assertNotIn("dimensions", dataset)
            self.assertNotIn("rgbSchema", dataset)

    def test_dataset_with_details(self):
        ctx = new_test_service_context()
        response = get_datasets(ctx, details=True, base_url="http://test")
        datasets = self._assert_datasets(response, expected_count=2)

        demo_dataset = None
        demo_1w_dataset = None
        for dataset in datasets:
            self.assertIsInstance(dataset, dict)
            self.assertIn("id", dataset)
            self.assertIn("title", dataset)
            self.assertIn("attributions", dataset)
            self.assertIn("variables", dataset)
            self.assertIn("dimensions", dataset)
            self.assertNotIn("rgbSchema", dataset)
            if dataset["id"] == "demo":
                demo_dataset = dataset
            if dataset["id"] == "demo-1w":
                demo_1w_dataset = dataset

        self.assertIsNotNone(demo_dataset)
        self.assertIsNotNone(demo_1w_dataset)
        self.assertEqual(["© by EU H2020 CyanoAlert project"], demo_dataset['attributions'])
        self.assertEqual(["© by Brockmann Consult GmbH 2020, "
                          "contains modified Copernicus Data 2019, processed by ESA"], demo_1w_dataset['attributions'])

    def test_dataset_with_details_and_rgb_schema(self):
        ctx = new_test_service_context('config-rgb.yml')
        response = get_datasets(ctx, details=True, base_url="http://test")
        datasets = self._assert_datasets(response, expected_count=1)
        dataset = datasets[0]
        self.assertIsInstance(dataset, dict)
        self.assertEqual({'varNames': ['conc_chl', 'conc_tsm', 'kd489'],
                          'normRanges': [(0.0, 24.0), (0.0, 100.0),
                                         (0.0, 6.0)]},
                         dataset.get("rgbSchema"))
        response = get_datasets(ctx, details=True, client='ol4',
                                base_url="http://test")
        datasets = self._assert_datasets(response, expected_count=1)
        dataset = datasets[0]
        self.assertIsInstance(dataset, dict)
        self.assertEqual({'varNames': ['conc_chl', 'conc_tsm', 'kd489'],
                          'normRanges': [(0.0, 24.0), (0.0, 100.0),
                                         (0.0, 6.0)],
                          'tileSourceOptions': {
                              'url': 'http://test/datasets/demo-rgb/vars/rgb/tiles/{z}/{x}/{y}.png',
                              'projection': 'EPSG:4326',
                              'tileGrid': {'extent': [0, 50, 5, 52.5],
                                           'origin': [0, 52.5],
                                           'resolutions': [0.01, 0.005,
                                                           0.0025],
                                           'sizes': [[2, 1], [4, 2], [8, 4]],
                                           'tileSize': [250, 250]}
                          }},
                         dataset.get("rgbSchema"))

    def test_dataset_with_point(self):
        ctx = new_test_service_context()

        response = get_datasets(ctx, point=(1.7, 51.2),
                                base_url="http://test")
        datasets = self._assert_datasets(response, expected_count=2)
        dataset = datasets[0]
        self.assertIsInstance(dataset, dict)
        self.assertIn("id", dataset)
        self.assertIn("title", dataset)
        self.assertNotIn("variables", dataset)
        self.assertNotIn("dimensions", dataset)

        response = get_datasets(ctx, point=(1.7, 58.0),
                                base_url="http://test")
        self._assert_datasets(response, expected_count=0)

    def test_dataset_with_point_and_details(self):
        ctx = new_test_service_context()

        response = get_datasets(ctx, point=(1.7, 51.2), details=True,
                                base_url="http://test")
        datasets = self._assert_datasets(response, expected_count=2)
        dataset = datasets[0]
        self.assertIsInstance(dataset, dict)
        self.assertIn("id", dataset)
        self.assertIn("title", dataset)
        self.assertIn("variables", dataset)
        self.assertIn("dimensions", dataset)

        response = get_datasets(ctx, point=(1.7, 58.0), details=True,
                                base_url="http://test")
        self._assert_datasets(response, 0)

    def test_get_colorbars(self):
        ctx = ServiceContext()

        response = get_color_bars(ctx, 'application/json')
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 40)
        self.assertEqual('[\n  [\n    "Perceptually Uniform Sequenti',
                         response[0:40])

        response = get_color_bars(ctx, 'text/html')
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 40)
        self.assertEqual('<!DOCTYPE html>\n<html lang="en">\n<head><',
                         response[0:40])

        with self.assertRaises(ServiceBadRequestError) as cm:
            get_color_bars(ctx, 'text/xml')
        self.assertEqual(400, cm.exception.status_code)
        self.assertEqual("Format 'text/xml' not supported for color bars",
                         cm.exception.reason)

    def _assert_datasets(self,
                         response: Any,
                         expected_count: Optional[int] = None):
        self.assertIsInstance(response, dict)
        datasets = response.get("datasets")
        self.assertIsInstance(datasets, list)
        if expected_count is not None:
            self.assertEqual(expected_count, len(datasets))
        return datasets
