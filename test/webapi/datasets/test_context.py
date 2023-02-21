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
from typing import Union, Mapping, Any

import pytest
import xarray as xr

from test.webapi.helpers import get_api_ctx
from test.webapi.helpers import get_server
from xcube.core.mldataset import MultiLevelDataset
from xcube.server.api import ApiError
from xcube.server.api import Context
from xcube.webapi.datasets.context import DatasetsContext


def get_datasets_ctx(
        server_config: Union[str, Mapping[str, Any]] = "config.yml"
) -> DatasetsContext:
    return get_api_ctx("datasets", DatasetsContext, server_config)


class DatasetsContextTest(unittest.TestCase):

    def test_ctx_ok(self):
        ctx = get_datasets_ctx()
        self.assertIsInstance(ctx.server_ctx, Context)
        self.assertIsInstance(ctx.auth_ctx, Context)
        self.assertIsInstance(ctx.places_ctx, Context)

    def test_get_dataset_and_variable(self):
        ctx = get_datasets_ctx()

        ds = ctx.get_dataset('demo')
        self.assertIsInstance(ds, xr.Dataset)

        ml_ds = ctx.get_ml_dataset('demo')
        self.assertIsInstance(ml_ds, MultiLevelDataset)
        self.assertIs(3, ml_ds.num_levels)
        self.assertIs(ds, ml_ds.get_dataset(0))

        for var_name in ('conc_chl', 'conc_tsm'):
            for z in range(ml_ds.num_levels):
                conc_chl_z = ctx.get_variable_for_z('demo', var_name, z)
                self.assertIsInstance(conc_chl_z, xr.DataArray)
            with self.assertRaises(ApiError.NotFound) as cm:
                ctx.get_variable_for_z('demo', var_name, 3)
            self.assertEqual(404, cm.exception.status_code)
            self.assertEqual(f'HTTP status 404: Variable "{var_name}"'
                             f' has no z-index 3 in dataset "demo"',
                             f'{cm.exception}')

        with self.assertRaises(ApiError.NotFound) as cm:
            ctx.get_variable_for_z('demo', 'conc_ys', 0)
        self.assertEqual(404, cm.exception.status_code)
        self.assertEqual('HTTP status 404:'
                         ' Variable "conc_ys" not found in dataset "demo"',
                         f'{cm.exception}')

        with self.assertRaises(ApiError.NotFound) as cm:
            ctx.get_dataset('demox')
        self.assertEqual(404, cm.exception.status_code)
        self.assertEqual('HTTP status 404: Dataset "demox" not found',
                         f'{cm.exception}')

        with self.assertRaises(ApiError.NotFound) as cm:
            ctx.get_dataset('demo', expected_var_names=['conc_ys'])
        self.assertEqual(404, cm.exception.status_code)
        self.assertEqual('HTTP status 404:'
                         ' Variable "conc_ys" not found in dataset "demo"',
                         f'{cm.exception}')

    def test_get_dataset_with_augmentation(self):
        ctx = get_datasets_ctx('config-aug.yml')

        ds = ctx.get_dataset('demo-aug')
        self.assertIsInstance(ds, xr.Dataset)

        ml_ds = ctx.get_ml_dataset('demo-aug')
        self.assertIsInstance(ml_ds, MultiLevelDataset)
        self.assertIs(3, ml_ds.num_levels)
        self.assertIs(ds, ml_ds.get_dataset(0))

        for var_name in ('conc_chl', 'conc_tsm',
                         'chl_tsm_sum', 'chl_category'):
            for z in range(ml_ds.num_levels):
                conc_chl_z = ctx.get_variable_for_z('demo-aug', var_name, z)
                self.assertIsInstance(conc_chl_z, xr.DataArray)

    def test_get_dataset_configs_from_stores(self):
        ctx = get_datasets_ctx('config-datastores.yml')

        dataset_configs_from_stores = ctx.get_dataset_configs_from_stores(
            ctx.get_data_store_pool()
        )
        self.assertIsNotNone(dataset_configs_from_stores)
        self.assertEqual(3, len(dataset_configs_from_stores))
        ids = [config['Identifier'] for config in dataset_configs_from_stores]
        self.assertEqual({'test~cube-1-250-250.levels',
                          'Cube-T5.zarr',
                          'Cube-T1.zarr'},
                         set(ids))

    def test_get_dataset_configs_with_duplicate_ids_from_stores(self):
        with self.assertRaises(ApiError.InvalidServerConfig) as sce:
            ctx = get_datasets_ctx('config-datastores-double-ids.yml')
        self.assertEqual('HTTP status 580:'
                         ' User-defined identifiers can only be assigned to '
                         'datasets with non-wildcard paths.',
                         f'{sce.exception}')

    def test_config_and_dataset_cache(self):
        ctx = get_datasets_ctx()
        self.assertNotIn('demo', ctx.dataset_cache)

        ctx.get_dataset('demo')
        self.assertIn('demo', ctx.dataset_cache)

        base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..")

        ctx = get_datasets_ctx(dict(
            base_dir=base_dir,
            Datasets=[
                dict(Identifier='demo',
                     Path="examples/serve/demo/cube.nc"),
                dict(Identifier='demo2',
                     Path="examples/serve/demo/cube.nc"),
            ]
        ))
        self.assertNotIn('demo', ctx.dataset_cache)
        self.assertNotIn('demo2', ctx.dataset_cache)

        ctx.get_dataset('demo2')
        self.assertNotIn('demo', ctx.dataset_cache)
        self.assertIn('demo2', ctx.dataset_cache)

        ctx = get_datasets_ctx(dict(
            Datasets=[
                dict(Identifier='demo2',
                     Path="examples/serve/demo/cube.nc"),
            ]
        ))
        self.assertNotIn('demo', ctx.dataset_cache)
        self.assertNotIn('demo2', ctx.dataset_cache)

    def test_get_color_mappings(self):
        ctx = get_datasets_ctx()
        color_mapping = ctx.get_color_mappings('demo-1w')
        self.assertEqual(
            {
                'conc_chl': {'ColorBar': 'plasma',
                             'ValueRange': [0.0, 24.0]},
                'conc_tsm': {'ColorBar': 'PuBuGn',
                             'ValueRange': [0.0, 100.0]},
                'kd489': {'ColorBar': 'jet',
                          'ValueRange': [0.0, 6.0]}
            },
            color_mapping
        )

    def test_get_color_mapping(self):
        ctx = get_datasets_ctx()
        cm = ctx.get_color_mapping('demo', 'conc_chl')
        self.assertEqual(('plasma', (0., 24.)), cm)
        cm = ctx.get_color_mapping('demo', 'conc_tsm')
        self.assertEqual(('PuBuGn', (0., 100.)), cm)
        cm = ctx.get_color_mapping('demo', 'kd489')
        self.assertEqual(('jet', (0., 6.)), cm)
        with self.assertRaises(ApiError.NotFound):
            ctx.get_color_mapping('demo', '_')

    def test_get_rgb_color_mapping(self):
        ctx = get_datasets_ctx()
        rgb_cm = ctx.get_rgb_color_mapping('demo')
        self.assertEqual(
            ([None, None, None], [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]),
            rgb_cm)
        rgb_cm = ctx.get_rgb_color_mapping('demo', norm_range=(1.0, 2.5))
        self.assertEqual(
            ([None, None, None], [(1.0, 2.5), (1.0, 2.5), (1.0, 2.5)]),
            rgb_cm)
        ctx = get_datasets_ctx('config-rgb.yml')
        rgb_cm = ctx.get_rgb_color_mapping('demo-rgb')
        self.assertEqual((['conc_chl', 'conc_tsm', 'kd489'],
                          [(0.0, 24.0), (0.0, 100.0), (0.0, 6.0)]), rgb_cm)

    def test_get_other_store_params_than_root(self):
        ctx = get_datasets_ctx()
        dataset_config = {
            'Identifier': 'two',
            'Title': 'Test 2',
            'FileSystem': 's3',
            'Anonymous': False,
            'Endpoint': 'https://s3.eu-central-1.amazonaws.com',
            'Path': 'xcube-examples/OLCI-SNS-RAW-CUBE-2.zarr',
            'Region': 'eu-central-1'
        }
        store_params = ctx._get_other_store_params_than_root(dataset_config)
        expected_dict = {
            'storage_options': {
                'anon': False,
                'client_kwargs': {
                    'endpoint_url': 'https://s3.eu-central-1.amazonaws.com',
                    'region_name': 'eu-central-1'
                }
            }
        }
        self.assertIsNotNone(store_params)
        self.assertEqual(expected_dict, store_params)

    def test_computed_ml_dataset_ok(self):
        ctx = get_datasets_ctx(server_config='config-class.yml')
        ds1 = ctx.get_ml_dataset("ds-1")
        ds2 = ctx.get_ml_dataset("ds-2")
        self.assertEqual(set(ds1.base_dataset.coords),
                         set(ds2.base_dataset.coords))
        self.assertEqual(set(ds1.base_dataset.data_vars),
                         set(ds2.base_dataset.data_vars))

    def test_computed_ml_dataset_call_fails(self):
        ctx = get_datasets_ctx(server_config='config-class.yml')
        with pytest.raises(
                ApiError.InvalidServerConfig,
                match="HTTP status 580:"
                      " Invalid in-memory dataset descriptor 'ds-3':"
                      " broken_ml_dataset_factory_1\\(\\)"
                      " takes 0 positional arguments but 1 was given"
        ):
            ctx.get_ml_dataset("ds-3")

    def test_computed_ml_dataset_illegal_return(self):
        ctx = get_datasets_ctx(server_config='config-class.yml')
        with pytest.raises(
                ApiError.InvalidServerConfig,
                match="Invalid in-memory dataset descriptor 'ds-4':"
                      " 'script:broken_ml_dataset_factory_2' must return"
                      " instance of xcube.core.mldataset.MultiLevelDataset,"
                      " but was <class 'xarray.core.dataset.Dataset'>"
        ):
            ctx.get_ml_dataset("ds-4")

    def test_interpolate_config_value(self):
        ctx = get_datasets_ctx()

        self.assertEqual(
            f"{ctx.base_dir}",
            ctx.eval_config_value(
                "${base_dir}"
            )
        )

        self.assertEqual(
            f"{ctx.base_dir}/test.yaml",
            ctx.eval_config_value(
                "${base_dir}/test.yaml"
            )
        )

        self.assertEqual(
            f"{ctx.base_dir}/configs/../test.yaml",
            ctx.eval_config_value(
                "${base_dir}/configs/../test.yaml"
            )
        )

        self.assertEqual(
            f"{ctx.base_dir}/test.yaml".replace("\\", "/"),
            ctx.eval_config_value(
                "${resolve_config_path('configs/../test.yaml')}"
            )
        )

        self.assertIs(
            ctx,
            ctx.eval_config_value(
                "${ctx}"
            )
        )

        self.assertEqual(13, ctx.eval_config_value(13))
        self.assertEqual(True, ctx.eval_config_value(True))

        self.assertEqual(
            [f"{ctx.base_dir}/test.yaml", 13, True],
            ctx.eval_config_value(
                ["${base_dir}/test.yaml", 13, True]
            )
        )

        self.assertEqual(
            dict(path=f"{ctx.base_dir}/test.yaml", count=13, check=True),
            ctx.eval_config_value(
                dict(path="${base_dir}/test.yaml", count=13, check=True)
            )
        )

    def test_tokenize_value(self):
        ctx = get_datasets_ctx()

        self.assertEqual(["Hallo!"],
                         list(ctx._tokenize_value("Hallo!")))

        self.assertEqual([('base_dir',)],
                         list(ctx._tokenize_value("${base_dir}")))

        self.assertEqual([('base_dir',), "/../test"],
                         list(ctx._tokenize_value("${base_dir}/../test")))

        self.assertEqual(["file://", ('base_dir',), "/test"],
                         list(ctx._tokenize_value("file://${base_dir}/test")))

        self.assertEqual([("resolve_config_path('../test')",)],
                         list(ctx._tokenize_value(
                             "${resolve_config_path('../test')}"
                         )))


class MaybeAssignStoreInstanceIdsTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.server = get_server('config.yml')

    def get_datasets_ctx(self, **config_delta) -> DatasetsContext:
        if config_delta:
            config = dict(self.server.ctx.config)
            config.update(config_delta)
            self.server.update(config)
        ctx = self.server.ctx.get_api_ctx("datasets", cls=DatasetsContext)
        self.assertIsInstance(ctx, DatasetsContext)
        return ctx

    def test_find_common_store(self):
        dataset_configs = [
            {
                'Identifier': 'z_0',
                'FileSystem': 'file',
                'Path': '/one/path/abc.zarr'
            },
            {
                'Identifier': 'z_1',
                'FileSystem': 'file',
                'Path': '/one/path/def.zarr'
            },
            {
                'Identifier': 'z_4',
                'FileSystem': 's3',
                'Path': '/one/path/mno.zarr'
            },
            {
                'Identifier': 'z_2',
                'FileSystem': 'file',
                'Path': '/another/path/ghi.zarr'
            },
            {
                'Identifier': 'z_3',
                'FileSystem': 'file',
                'Path': '/one/more/path/jkl.zarr'
            },
            {
                'Identifier': 'z_5',
                'FileSystem': 's3',
                'Path': '/one/path/pqr.zarr'
            },
            {
                'Identifier': 'z_6',
                'FileSystem': 'file',
                'Path': '/one/path/stu.zarr'
            },
            {
                'Identifier': 'z_7',
                'FileSystem': 'file',
                'Path': '/one/more/path/vwx.zarr'
            },
        ]

        ctx = self.get_datasets_ctx(Datasets=dataset_configs)
        # ctx.config['Datasets'] = dataset_configs
        adjusted_dataset_configs = ctx.get_dataset_configs()

        expected_dataset_configs = [
            {
                'Identifier': 'z_0',
                'FileSystem': 'file',
                'Path': 'path/abc.zarr',
                'StoreInstanceId': 'file_2'
            },
            {
                'Identifier': 'z_1',
                'FileSystem': 'file',
                'Path': 'path/def.zarr',
                'StoreInstanceId': 'file_2'
            },
            {
                'Identifier': 'z_4',
                'FileSystem': 's3',
                'Path': 'mno.zarr',
                'StoreInstanceId': 's3_1'
            },
            {
                'Identifier': 'z_2',
                'FileSystem': 'file',
                'Path': 'ghi.zarr',
                'StoreInstanceId': 'file_1'
            },
            {
                'Identifier': 'z_3',
                'FileSystem': 'file',
                'Path': 'more/path/jkl.zarr',
                'StoreInstanceId': 'file_2'
            },
            {
                'Identifier': 'z_5',
                'FileSystem': 's3',
                'Path': 'pqr.zarr',
                'StoreInstanceId': 's3_1'
            },
            {
                'Identifier': 'z_6',
                'FileSystem': 'file',
                'Path': 'path/stu.zarr',
                'StoreInstanceId': 'file_2'
            },
            {
                'Identifier': 'z_7',
                'FileSystem': 'file',
                'Path': 'more/path/vwx.zarr',
                'StoreInstanceId': 'file_2'
            },
        ]
        self.assertEqual(expected_dataset_configs, adjusted_dataset_configs)

    def test_with_instance_id(self):
        dataset_config = {'Identifier': 'zero',
                          'Title': 'Test 0',
                          'FileSystem': 'file',
                          'Path': 'some.zarr'}

        ctx = self.get_datasets_ctx(Datasets=[dataset_config])
        dataset_config = ctx.get_dataset_configs()[0]

        self.assertEqual({'FileSystem': 'file',
                          'Identifier': 'zero',
                          'Path': 'some.zarr',
                          'StoreInstanceId': 'file_1',
                          'Title': 'Test 0'},
                         dataset_config)

    def test_file(self):
        dataset_config = {'Identifier': 'one',
                          'Title': 'Test 1',
                          'FileSystem': 'file',
                          'Path': 'cube-1-250-250.zarr'}

        ctx = self.get_datasets_ctx(Datasets=[dataset_config])
        dataset_config = ctx.get_dataset_configs()[0]

        self.assertEqual({'Identifier', 'Title', 'FileSystem', 'Path',
                          'StoreInstanceId'},
                         set(dataset_config.keys()))
        self.assertEqual('one',
                         dataset_config['Identifier'])
        self.assertEqual('Test 1', dataset_config['Title'])
        self.assertEqual('file', dataset_config['FileSystem'])
        self.assertEqual('cube-1-250-250.zarr', dataset_config["Path"])
        self.assertEqual('file_1', dataset_config['StoreInstanceId'])

    def test_local(self):
        # this test tests backwards compatibility.
        # TODO please remove when support for file systems 'local' and 'obs'
        # has ended
        dataset_config = {'Identifier': 'one',
                          'Title': 'Test 1',
                          'FileSystem': 'local',
                          'Path': 'cube-1-250-250.zarr'}

        ctx = self.get_datasets_ctx(Datasets=[dataset_config])
        dataset_config = ctx.get_dataset_configs()[0]

        self.assertEqual({'Identifier', 'Title', 'FileSystem', 'Path',
                          'StoreInstanceId'},
                         set(dataset_config.keys()))
        self.assertEqual('one',
                         dataset_config['Identifier'])
        self.assertEqual('Test 1', dataset_config['Title'])
        self.assertEqual('local', dataset_config['FileSystem'])
        self.assertEqual('cube-1-250-250.zarr', dataset_config["Path"])
        self.assertEqual('file_1', dataset_config['StoreInstanceId'])

    def test_s3(self):
        dataset_config = {'Identifier': 'two',
                          'Title': 'Test 2',
                          'FileSystem': 's3',
                          'Endpoint': 'https://s3.eu-central-1.amazonaws.com',
                          'Path': 'xcube-examples/OLCI-SNS-RAW-CUBE-2.zarr',
                          'Region': 'eu-central-1'}

        ctx = self.get_datasets_ctx(Datasets=[dataset_config])
        dataset_config = ctx.get_dataset_configs()[0]

        self.assertEqual({'Identifier', 'Title', 'FileSystem', 'Endpoint',
                          'Path', 'Region', 'StoreInstanceId'},
                         set(dataset_config.keys()))
        self.assertEqual('two', dataset_config['Identifier'])
        self.assertEqual('Test 2', dataset_config['Title'])
        self.assertEqual('s3', dataset_config['FileSystem'])
        self.assertEqual('https://s3.eu-central-1.amazonaws.com',
                         dataset_config['Endpoint'])
        self.assertEqual('OLCI-SNS-RAW-CUBE-2.zarr', dataset_config['Path'])
        self.assertEqual('eu-central-1', dataset_config['Region'])
        self.assertEqual('s3_1', dataset_config['StoreInstanceId'])

    def test_obs(self):
        # this test tests backwards compatibility.
        # TODO please remove when support for file systems 'local' and 'obs'
        # has ended
        dataset_config = {'Identifier': 'two',
                          'Title': 'Test 2',
                          'FileSystem': 'obs',
                          'Endpoint': 'https://s3.eu-central-1.amazonaws.com',
                          'Path': 'xcube-examples/OLCI-SNS-RAW-CUBE-2.zarr',
                          'Region': 'eu-central-1'}

        ctx = self.get_datasets_ctx(Datasets=[dataset_config])
        dataset_config = ctx.get_dataset_configs()[0]

        self.assertEqual({'Identifier', 'Title', 'FileSystem', 'Endpoint',
                          'Path', 'Region', 'StoreInstanceId'},
                         set(dataset_config.keys()))
        self.assertEqual('two', dataset_config['Identifier'])
        self.assertEqual('Test 2', dataset_config['Title'])
        self.assertEqual('obs', dataset_config['FileSystem'])
        self.assertEqual('https://s3.eu-central-1.amazonaws.com',
                         dataset_config['Endpoint'])
        self.assertEqual('OLCI-SNS-RAW-CUBE-2.zarr', dataset_config['Path'])
        self.assertEqual('eu-central-1', dataset_config['Region'])
        self.assertEqual('s3_1', dataset_config['StoreInstanceId'])

    def test_memory(self):
        dataset_config = {'Identifier': 'three',
                          'Title': 'Test 3',
                          'FileSystem': 'memory',
                          'Path': 'calc.py'}
        dataset_config_copy = dataset_config.copy()

        ctx = self.get_datasets_ctx(Datasets=[dataset_config])
        dataset_config = ctx.get_dataset_configs()[0]

        self.assertEqual(dataset_config_copy, dataset_config)

    def test_missing_file_system(self):
        dataset_config = {'Identifier': 'five',
                          'Title': 'Test 5',
                          'Path': 'cube-1-250-250.zarr'}

        ctx = self.get_datasets_ctx(Datasets=[dataset_config])
        dataset_config = ctx.get_dataset_configs()[0]

        self.assertEqual({'Identifier', 'Title', 'Path', 'StoreInstanceId'},
                         set(dataset_config.keys()))
        self.assertEqual('five', dataset_config['Identifier'])
        self.assertEqual('Test 5', dataset_config['Title'])
        self.assertEqual('cube-1-250-250.zarr', dataset_config['Path'])
        self.assertEqual('file_1', dataset_config['StoreInstanceId'])

    def test_invalid_file_system(self):
        dataset_config = {'Identifier': 'five',
                          'Title': 'Test 5a',
                          'FileSystem': 'invalid',
                          'Path': 'cube-1-250-250.zarr'}
        with self.assertRaises(ValueError) as cm:
            self.get_datasets_ctx(Datasets=[dataset_config])
        self.assertRegexpMatches(
            f'{cm.exception}',
            'Invalid server configuration:\n.*'
        )

    def test_local_store_already_existing(self):
        dataset_config_1 = {'Identifier': 'six',
                            'Title': 'Test 6',
                            'FileSystem': 'file',
                            'Path': 'cube-1-250-250.zarr'}
        dataset_config_2 = {'Identifier': 'six_a',
                            'Title': 'Test 6 a',
                            'FileSystem': 'file',
                            'Path': 'cube-5-100-200.zarr'}

        ctx = self.get_datasets_ctx(Datasets=[dataset_config_1,
                                              dataset_config_2])
        dataset_configs = ctx.get_dataset_configs()

        self.assertEqual(dataset_configs[0]['StoreInstanceId'],
                         dataset_configs[1]['StoreInstanceId'])

    def test_s3_store_already_existing(self):
        dataset_config_1 = {
            'Identifier': 'seven',
            'Title': 'Test 7',
            'FileSystem': 's3',
            'Endpoint': 'https://s3.eu-central-1.amazonaws.com',
            'Path': 'xcube-examples/OLCI-SNS-RAW-CUBE-2.zarr',
            'Region': 'eu-central-1'
        }

        dataset_config_2 = {
            'Identifier': 'seven_a',
            'Title': 'Test 7 a',
            'FileSystem': 's3',
            'Endpoint': 'https://s3.eu-central-1.amazonaws.com',
            'Path': 'xcube-examples/OLCI-SNS-RAW-CUBE-3.zarr',
            'Region': 'eu-central-1'
        }

        ctx = self.get_datasets_ctx(Datasets=[dataset_config_1,
                                              dataset_config_2])
        dataset_configs = ctx.get_dataset_configs()

        self.assertEqual(dataset_configs[0]['StoreInstanceId'],
                         dataset_configs[1]['StoreInstanceId'])

    def test_mix_of_absolute_and_relative_paths(self):
        configs = [
            {
                'Identifier': 'z_0',
                'FileSystem': 'file',
                'Path': '/path/abc.zarr'
            },
            {
                'Identifier': 'z_1',
                'FileSystem': 'file',
                'Path': 'def.zarr'
            },
            {
                'Identifier': 'z_2',
                'FileSystem': 'file',
                'Path': 'relative/ghi.zarr'
            }
        ]

        ctx = self.get_datasets_ctx(Datasets=configs)
        dataset_configs = ctx.get_dataset_configs()

        expected_dataset_configs = [
            {
                'Identifier': 'z_0',
                'FileSystem': 'file',
                'Path': 'abc.zarr',
                'StoreInstanceId': 'file_1'
            },
            {
                'Identifier': 'z_1',
                'FileSystem': 'file',
                'Path': 'def.zarr',
                'StoreInstanceId': 'file_3'
            },
            {
                'Identifier': 'z_2',
                'FileSystem': 'file',
                'Path': 'ghi.zarr',
                'StoreInstanceId': 'file_2'
            }
        ]

        self.assertEqual(expected_dataset_configs, dataset_configs)
