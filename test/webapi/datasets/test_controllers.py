# The MIT License (MIT)
# Copyright (c) 2022 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import os.path
import unittest
from typing import Any, Optional

from test.webapi.helpers import get_api_ctx
from xcube.core.new import new_cube
from xcube.server.api import ApiError
from xcube.webapi.datasets.context import DatasetsContext
from xcube.webapi.datasets.controllers import filter_variable_names
from xcube.webapi.datasets.controllers import find_dataset_places
from xcube.webapi.datasets.controllers import get_color_bars
from xcube.webapi.datasets.controllers import get_dataset
from xcube.webapi.datasets.controllers import get_datasets
from xcube.webapi.datasets.controllers import get_legend
from xcube.webapi.datasets.controllers import get_time_chunk_size


def get_datasets_ctx(server_config=None) -> DatasetsContext:
    return get_api_ctx("datasets", DatasetsContext, server_config)


class DatasetsControllerTestBase(unittest.TestCase):

    def assertDatasetsOk(self,
                         response: Any,
                         expected_count: Optional[int] = None):
        self.assertIsInstance(response, dict)
        datasets = response.get("datasets")
        self.assertIsInstance(datasets, list)
        if expected_count is not None:
            self.assertEqual(expected_count, len(datasets))
        return datasets


class DatasetsControllerTest(DatasetsControllerTestBase):

    def test_datasets(self):
        response = get_datasets(get_datasets_ctx())
        datasets = self.assertDatasetsOk(response, expected_count=2)
        for dataset in datasets:
            self.assertIsInstance(dataset, dict)
            self.assertIn("id", dataset)
            self.assertIn("title", dataset)
            self.assertNotIn("variables", dataset)
            self.assertNotIn("dimensions", dataset)
            self.assertNotIn("rgbSchema", dataset)

    def test_dataset_with_details(self):
        response = get_datasets(get_datasets_ctx(),
                                details=True,
                                base_url="http://test")
        datasets = self.assertDatasetsOk(response, expected_count=2)

        demo_dataset = None
        demo_1w_dataset = None
        for dataset in datasets:
            self.assertIsInstance(dataset, dict)
            self.assertIsInstance(dataset.get("id"), str)
            self.assertIsInstance(dataset.get("title"), str)
            self.assertIsInstance(dataset.get("attributions"), (tuple, list))
            self.assertIsInstance(dataset.get("variables"), (tuple, list))
            self.assertIsInstance(dataset.get("dimensions"), (tuple, list))
            self.assertIsInstance(dataset.get("bbox"), (tuple, list))
            self.assertIsInstance(dataset.get("geometry"), dict)
            self.assertIsInstance(dataset.get("spatialRef"), str)
            self.assertNotIn("rgbSchema", dataset)
            if dataset["id"] == "demo":
                demo_dataset = dataset
            if dataset["id"] == "demo-1w":
                demo_1w_dataset = dataset

        self.assertIsNotNone(demo_dataset)
        self.assertIsNotNone(demo_1w_dataset)
        self.assertEqual(['© by EU H2020 CyanoAlert project'],
                         demo_dataset['attributions'])
        self.assertEqual(["© by Brockmann Consult GmbH 2020, "
                          "contains modified Copernicus Data 2019, "
                          "processed by ESA"],
                         demo_1w_dataset['attributions'])

    def test_dataset_with_point(self):
        response = get_datasets(get_datasets_ctx(),
                                point=(1.7, 51.2),
                                base_url="http://test")
        datasets = self.assertDatasetsOk(response, expected_count=2)
        dataset = datasets[0]
        self.assertIsInstance(dataset, dict)
        self.assertIn("id", dataset)
        self.assertIn("title", dataset)
        self.assertNotIn("variables", dataset)
        self.assertNotIn("dimensions", dataset)

        response = get_datasets(get_datasets_ctx(),
                                point=(1.7, 58.0),
                                base_url="http://test")
        self.assertDatasetsOk(response, expected_count=0)

    def test_dataset_with_point_and_details(self):
        response = get_datasets(get_datasets_ctx(),
                                point=(1.7, 51.2),
                                details=True,
                                base_url="http://test")
        datasets = self.assertDatasetsOk(response, expected_count=2)
        dataset = datasets[0]
        self.assertIsInstance(dataset, dict)
        self.assertIn("id", dataset)
        self.assertIn("title", dataset)
        self.assertIn("variables", dataset)
        self.assertIn("dimensions", dataset)

        response = get_datasets(get_datasets_ctx(),
                                point=(1.7, 58.0),
                                details=True,
                                base_url="http://test")
        self.assertDatasetsOk(response, 0)

    def test_get_colorbars(self):
        response = get_color_bars(get_datasets_ctx(), 'application/json')
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 40)
        self.assertEqual('[\n  [\n    "Perceptually Uniform Sequenti',
                         response[0:40])

        response = get_color_bars(get_datasets_ctx(), 'text/html')
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 40)
        self.assertEqual('<!DOCTYPE html>\n<html lang="en">\n<head><',
                         response[0:40])

        with self.assertRaises(ApiError.BadRequest) as cm:
            get_color_bars(get_datasets_ctx(), 'text/xml')
        self.assertEqual(400, cm.exception.status_code)
        self.assertEqual("HTTP status 400:"
                         " Format 'text/xml' not supported for colormaps",
                         f'{cm.exception}')

    def test_dataset_with_details_and_rgb_schema(self):
        response = get_datasets(get_datasets_ctx('config-rgb.yml'),
                                details=True,
                                base_url="http://test")
        datasets = self.assertDatasetsOk(response, expected_count=1)
        dataset = datasets[0]
        self.assertIsInstance(dataset, dict)
        self.assertEqual(
            {
                'varNames': ['conc_chl', 'conc_tsm', 'kd489'],
                'normRanges': [(0.0, 24.0), (0.0, 100.0),
                               (0.0, 6.0)],
                'tileUrl': 'http://test/datasets/demo-rgb/vars/rgb/'
                           'tiles2/{z}/{y}/{x}',
                'tileLevelMin': 7,
                'tileLevelMax': 9,
            },
            dataset.get("rgbSchema")
        )

        response = get_datasets(get_datasets_ctx('config-rgb.yml'),
                                details=True,
                                base_url="http://test")
        datasets = self.assertDatasetsOk(response, expected_count=1)
        dataset = datasets[0]
        self.assertIsInstance(dataset, dict)
        self.assertEqual(
            {
                'varNames': ['conc_chl', 'conc_tsm', 'kd489'],
                'normRanges': [(0.0, 24.0), (0.0, 100.0),
                               (0.0, 6.0)],
                'tileUrl': 'http://test/datasets/demo-rgb/vars/rgb/tiles2/{z}/{y}/{x}',
                'tileLevelMin': 7,
                'tileLevelMax': 9,
            },
            dataset.get("rgbSchema")
        )

    def test_find_dataset_features(self):
        places = find_dataset_places(get_datasets_ctx(),
                                     "all",
                                     "demo",
                                     "http://localhost:8080")
        self.assertPlaceGroupOk(places, 3, {'0', '1', '2'})

    def assertPlaceGroupOk(self, feature_collection, expected_count,
                           expected_ids):
        self.assertIsInstance(feature_collection, dict)
        self.assertIn("type", feature_collection)
        self.assertEqual("FeatureCollection", feature_collection["type"])
        self.assertIn("features", feature_collection)
        features = feature_collection["features"]
        self.assertIsInstance(features, list)
        self.assertEqual(expected_count, len(features))
        actual_ids = {f["id"] for f in features if "id" in f}
        self.assertEqual(expected_ids, actual_ids)


class DatasetsAuthControllerTest(DatasetsControllerTestBase):
    @staticmethod
    def get_config():
        data_store_root = os.path.normpath(os.path.join(
            os.path.dirname(__file__),
            '..', '..', '..', 'examples', 'serve', 'demo'
        ))

        return dict(
            base_dir=data_store_root,
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
                            Identifier='local_base_id',
                            Title='A local base dataset',
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
                            Title='A remote base dataset',
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
                    Path='resample_in_time.py',
                    Function='compute_dataset',
                    InputDatasets=['local_base_id'],
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
                    Path='resample_in_time.py',
                    Function='compute_dataset',
                    InputDatasets=['remote~OLCI-SNS-RAW-CUBE-2.zarr'],
                    InputParameters=dict(
                        period='1W',
                        incl_stdev=True
                    ),
                ),
            ],
        )

    def test_unauthorized_access(self):
        ctx = get_datasets_ctx(self.get_config())
        granted_scopes = None
        response = get_datasets(ctx, granted_scopes=granted_scopes)
        datasets = self.assertDatasetsOk(response)
        datasets_dict = {ds['id']: ds for ds in datasets}
        self.assertEqual(
            {
                'local_base_id',
                'local_base_1w',
                # Not selected, because they require authorisation
                # 'remote~OLCI-SNS-RAW-CUBE-2.zarr',
                # 'remote_base_1w',
            },
            set(datasets_dict)
        )
        dataset_titles_dict = {ds['title']: ds for ds in datasets}
        self.assertEqual(
            {
                'local_base_1w',
                'A local base dataset',
                # Not selected, because they require authorisation
                # 'remote_base_1w',
                # 'A remote base dataset',
            },
            set(dataset_titles_dict)
        )
        dataset = get_dataset(ctx, 'local_base_id')
        self.assertIn('variables', dataset)
        var_dict = {v['name']: v for v in dataset['variables']}
        self.assertEqual({'c2rcc_flags',
                          'conc_tsm',
                          'kd489',
                          'conc_chl',
                          'quality_flags'},
                         set(var_dict.keys()))

        dataset = get_dataset(ctx, 'local_base_1w')
        self.assertIn('variables', dataset)
        var_dict = {v['name']: v for v in dataset['variables']}
        self.assertEqual({'c2rcc_flags',
                          'c2rcc_flags_stdev',
                          'conc_chl',
                          'conc_chl_stdev',
                          'conc_tsm',
                          'conc_tsm_stdev',
                          'kd489',
                          'kd489_stdev',
                          'quality_flags',
                          'quality_flags_stdev'},
                         set(var_dict.keys()))

        with self.assertRaises(ApiError.Unauthorized) as cm:
            get_dataset(ctx, 'remote_base_1w')
        self.assertEqual(
            'HTTP status 401: Missing permission'
            ' "read:dataset:remote_base_1w"',
            f'{cm.exception}'
        )

        with self.assertRaises(ApiError.Unauthorized) as cm:
            get_dataset(ctx, 'remote~OLCI-SNS-RAW-CUBE-2.zarr')
        self.assertEqual(
            'HTTP status 401: Missing permission'
            ' "read:dataset:remote~OLCI-SNS-RAW-CUBE-2.zarr"',
            f'{cm.exception}'
        )

    def test_authorized_access_with_joker_scopes(self):
        ctx = get_datasets_ctx(self.get_config())
        granted_scopes = {
            'read:dataset:*',
            'read:variable:*'
        }
        response = get_datasets(ctx, granted_scopes=granted_scopes)
        datasets = self.assertDatasetsOk(response)
        dataset_ids_dict = {ds['id']: ds for ds in datasets}
        self.assertEqual(
            {
                'local_base_1w',
                'local_base_id',
                'remote_base_1w',
                'remote~OLCI-SNS-RAW-CUBE-2.zarr',
            },
            set(dataset_ids_dict)
        )
        dataset_titles_dict = {ds['title']: ds for ds in datasets}
        self.assertEqual(
            {
                'local_base_1w',
                'A local base dataset',
                'remote_base_1w',
                'A remote base dataset',
            },
            set(dataset_titles_dict)
        )

    def test_authorized_access_with_specific_scopes(self):
        ctx = get_datasets_ctx(self.get_config())
        granted_scopes = {
            'read:dataset:remote*',
            'read:variable:*'
        }
        response = get_datasets(ctx, granted_scopes=granted_scopes)
        datasets = self.assertDatasetsOk(response)
        dataset_ids_dict = {ds['id']: ds for ds in datasets}
        self.assertEqual(
            {
                # Not selected, because they are substitutes
                # 'local_base_1w',
                # 'local_base_id',
                'remote_base_1w',
                'remote~OLCI-SNS-RAW-CUBE-2.zarr',
            },
            set(dataset_ids_dict)
        )
        dataset_titles_dict = {ds['title']: ds for ds in datasets}
        self.assertEqual(
            {
                # Not selected, because they are substitutes
                # 'local_base_1w',
                # 'A local base dataset',
                'remote_base_1w',
                'A remote base dataset',
            },
            set(dataset_titles_dict)
        )


class TimeChunkSizeTest(unittest.TestCase):

    @staticmethod
    def _get_cube(time_chunk_size: int = None):
        ts_ds = new_cube(time_periods=10, variables=dict(CHL=10.2))
        if time_chunk_size:
            ts_ds = ts_ds.assign(
                CHL=ts_ds.CHL.chunk(dict(time=time_chunk_size))
            )
        return ts_ds

    def test_get_time_chunk_size_is_ok(self):
        ts_ds = self._get_cube(time_chunk_size=1)
        self.assertEqual(1, get_time_chunk_size(ts_ds, 'CHL', 'ds.zarr'))

        ts_ds = self._get_cube(time_chunk_size=3)
        self.assertEqual(3, get_time_chunk_size(ts_ds, 'CHL', 'ds.zarr'))

        ts_ds = self._get_cube(time_chunk_size=5)
        self.assertEqual(5, get_time_chunk_size(ts_ds, 'CHL', 'ds.zarr'))

        ts_ds = self._get_cube(time_chunk_size=10)
        self.assertEqual(10, get_time_chunk_size(ts_ds, 'CHL', 'ds.zarr'))

    def test_get_time_chunk_size_fails(self):
        # TS dataset not given
        self.assertEqual(None, get_time_chunk_size(None, 'CHL', 'ds.zarr'))
        # Variable not found
        ts_ds = self._get_cube(time_chunk_size=5)
        self.assertEqual(None, get_time_chunk_size(ts_ds, 'MCI', 'ds.zarr'))
        # Time is not chunked
        ts_ds = self._get_cube(time_chunk_size=None)
        self.assertEqual(None, get_time_chunk_size(ts_ds, 'CHL', 'ds.zarr'))
        # Variable has no dimension "time"
        ts_ds = self._get_cube(time_chunk_size=5)
        ts_ds['CHL0'] = ts_ds.CHL.isel(time=0)
        self.assertEqual(None, get_time_chunk_size(ts_ds, 'CHL0', 'ds.zarr'))


class DatasetLegendTest(unittest.TestCase):
    def test_get_legend(self):
        ctx = get_datasets_ctx('config.yml')

        image = get_legend(ctx, 'demo', 'conc_chl', {})
        self.assertEqual("<class 'bytes'>", str(type(image)))

        # This is fine, because we fall back to "viridis".
        image = get_legend(ctx, 'demo', 'conc_chl', dict(cbar='sun-shine'))
        self.assertEqual("<class 'bytes'>", str(type(image)))

        with self.assertRaises(ApiError.BadRequest) as cm:
            get_legend(ctx, 'demo', 'conc_chl', dict(vmin='sun-shine'))
        self.assertEqual('HTTP status 400:'
                         ' Invalid color legend parameter(s)',
                         f'{cm.exception}')

        with self.assertRaises(ApiError.BadRequest) as cm:
            get_legend(ctx, 'demo', 'conc_chl', dict(width='sun-shine'))
        self.assertEqual('HTTP status 400:'
                         ' Invalid color legend parameter(s)',
                         f'{cm.exception}')


class DatasetsVariableNamesFilter(unittest.TestCase):
    names = [
        "c2rcc_flags",
        "quality_flags",
        "conc_tsm",
        "kd489",
        "conc_chl",
        "chl_category",
        "chl_tsm_sum",
    ]

    def test_same(self):
        filtered_names = filter_variable_names(self.names, ["*"])
        self.assertEqual(self.names, filtered_names)

    def test_ordered(self):

        filtered_names = filter_variable_names(self.names, [
            "conc_chl",
            "conc_tsm",
            "chl_tsm_sum",
            "*",
        ])

        self.assertEqual(len(self.names), len(filtered_names))
        self.assertEqual(["conc_chl",
                          "conc_tsm",
                          "chl_tsm_sum"],
                         filtered_names[0:3])

    def test_subset(self):
        filtered_names = filter_variable_names(self.names, [
            "conc_*",
            "chl_*",
        ])

        self.assertEqual(4, len(filtered_names))
        self.assertEqual({"conc_chl", "conc_tsm"},
                         set(filtered_names[0:2]))
        self.assertEqual({"chl_tsm_sum", "chl_category"},
                         set(filtered_names[2:4]))
