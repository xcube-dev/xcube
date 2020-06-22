import os.path
import unittest

import xarray as xr

from test.webapi.helpers import new_test_service_context
from xcube.core.mldataset import MultiLevelDataset
from xcube.webapi.context import normalize_prefix
from xcube.webapi.errors import ServiceResourceNotFoundError


class ServiceContextTest(unittest.TestCase):
    def test_get_dataset_and_variable(self):
        ctx = new_test_service_context()

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
            with self.assertRaises(ServiceResourceNotFoundError) as cm:
                ctx.get_variable_for_z('demo', var_name, 3)
            self.assertEqual(404, cm.exception.status_code)
            self.assertEqual(f'Variable "{var_name}" has no z-index 3 in dataset "demo"', cm.exception.reason)

        with self.assertRaises(ServiceResourceNotFoundError) as cm:
            ctx.get_variable_for_z('demo', 'conc_ys', 0)
        self.assertEqual(404, cm.exception.status_code)
        self.assertEqual('Variable "conc_ys" not found in dataset "demo"', cm.exception.reason)

        with self.assertRaises(ServiceResourceNotFoundError) as cm:
            ctx.get_dataset('demox')
        self.assertEqual(404, cm.exception.status_code)
        self.assertEqual('Dataset "demox" not found', cm.exception.reason)

        with self.assertRaises(ServiceResourceNotFoundError) as cm:
            ctx.get_dataset('demo', expected_var_names=['conc_ys'])
        self.assertEqual(404, cm.exception.status_code)
        self.assertEqual('Variable "conc_ys" not found in dataset "demo"', cm.exception.reason)

    def test_get_dataset_with_augmentation(self):
        ctx = new_test_service_context(config_file_name='config-aug.yml')

        ds = ctx.get_dataset('demo-aug')
        self.assertIsInstance(ds, xr.Dataset)

        ml_ds = ctx.get_ml_dataset('demo-aug')
        self.assertIsInstance(ml_ds, MultiLevelDataset)
        self.assertIs(3, ml_ds.num_levels)
        self.assertIs(ds, ml_ds.get_dataset(0))

        for var_name in ('conc_chl', 'conc_tsm', 'chl_tsm_sum', 'chl_category'):
            for z in range(ml_ds.num_levels):
                conc_chl_z = ctx.get_variable_for_z('demo-aug', var_name, z)
                self.assertIsInstance(conc_chl_z, xr.DataArray)

    def test_config_and_dataset_cache(self):
        ctx = new_test_service_context()
        self.assertNotIn('demo', ctx.dataset_cache)

        ctx.get_dataset('demo')
        self.assertIn('demo', ctx.dataset_cache)

        ctx.config = dict(Datasets=[
            dict(Identifier='demo',
                 Path="../../../../examples/serve/demo/cube.nc"),
            dict(Identifier='demo2',
                 Path="../../../../examples/serve/demo/cube.nc"),
        ])
        self.assertNotIn('demo', ctx.dataset_cache)
        self.assertNotIn('demo2', ctx.dataset_cache)

        ctx.get_dataset('demo2')
        self.assertNotIn('demo', ctx.dataset_cache)
        self.assertIn('demo2', ctx.dataset_cache)

        ctx.config = dict(Datasets=[
            dict(Identifier='demo2',
                 Path="../../../../examples/serve/demo/cube.nc"),
        ])
        self.assertNotIn('demo', ctx.dataset_cache)
        self.assertNotIn('demo2', ctx.dataset_cache)

    def test_get_s3_bucket_mapping(self):
        ctx = new_test_service_context()
        bucket_mapping = ctx.get_s3_bucket_mapping()
        self.assertEqual(['demo'],
                         list(bucket_mapping.keys()))
        path = bucket_mapping['demo']
        self.assertTrue(os.path.isabs(path))
        self.assertTrue(path.replace('\\', '/').endswith('examples/serve/demo/cube-1-250-250.zarr'))

    def test_get_color_mapping(self):
        ctx = new_test_service_context()
        cm = ctx.get_color_mapping('demo', 'conc_chl')
        self.assertEqual(('plasma', (0., 24.)), cm)
        cm = ctx.get_color_mapping('demo', 'conc_tsm')
        self.assertEqual(('PuBuGn', (0., 100.)), cm)
        cm = ctx.get_color_mapping('demo', 'kd489')
        self.assertEqual(('jet', (0., 6.)), cm)
        with self.assertRaises(ServiceResourceNotFoundError):
            ctx.get_color_mapping('demo', '_')

    def test_get_rgb_color_mapping(self):
        ctx = new_test_service_context()
        rgb_cm = ctx.get_rgb_color_mapping('demo')
        self.assertEqual(([None, None, None], [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]), rgb_cm)
        rgb_cm = ctx.get_rgb_color_mapping('demo', norm_range=(1.0, 2.5))
        self.assertEqual(([None, None, None], [(1.0, 2.5), (1.0, 2.5), (1.0, 2.5)]), rgb_cm)
        ctx = new_test_service_context('config-rgb.yml')
        rgb_cm = ctx.get_rgb_color_mapping('demo-rgb')
        self.assertEqual((['conc_chl', 'conc_tsm', 'kd489'], [(0.0, 24.0), (0.0, 100.0), (0.0, 6.0)]), rgb_cm)

    def test_get_style(self):
        ctx = new_test_service_context()
        style = ctx.get_style('demo-1w')
        self.assertEqual({'Identifier': 'default',
                          'ColorMappings': {'conc_chl': {'ColorBar': 'plasma',
                                                         'ValueRange': [0.0, 24.0]},
                                            'conc_tsm': {'ColorBar': 'PuBuGn',
                                                         'ValueRange': [0.0, 100.0]},
                                            'kd489': {'ColorBar': 'jet', 'ValueRange': [0.0, 6.0]}}},
                         style)

    def test_get_global_place_groups(self):
        ctx = new_test_service_context()
        place_groups = ctx.get_global_place_groups("http://localhost:9090", load_features=False)
        self.assertIsInstance(place_groups, list)
        self.assertEqual(2, len(place_groups))
        for place_group in place_groups:
            self.assertIn("type", place_group)
            self.assertIn("id", place_group)
            self.assertIn("title", place_group)
            self.assertIn("propertyMapping", place_group)
            self.assertIn("sourcePaths", place_group)
            self.assertIn("sourceEncoding", place_group)
            self.assertIn("features", place_group)
        place_group = place_groups[0]
        self.assertEqual('inside-cube', place_group['id'])
        self.assertEqual('Points inside the cube', place_group['title'])
        self.assertEqual(None, place_group['features'])
        place_group = place_groups[1]
        self.assertEqual('outside-cube', place_group['id'])
        self.assertEqual('Points outside the cube', place_group['title'])
        self.assertEqual(None, place_group['features'])

        place_groups = ctx.get_global_place_groups("http://localhost:9090", load_features=True)
        self.assertIsInstance(place_groups, list)
        self.assertEqual(2, len(place_groups))
        for place_group in place_groups:
            self.assertIn("type", place_group)
            self.assertIn("id", place_group)
            self.assertIn("title", place_group)
            self.assertIn("propertyMapping", place_group)
            self.assertIn("sourcePaths", place_group)
            self.assertIn("sourceEncoding", place_group)
            self.assertIn("features", place_group)
        place_group = place_groups[0]
        self.assertEqual('inside-cube', place_group['id'])
        self.assertEqual('Points inside the cube', place_group['title'])
        self.assertIsNotNone(place_group['features'])
        place_group = place_groups[1]
        self.assertEqual('outside-cube', place_group['id'])
        self.assertEqual('Points outside the cube', place_group['title'])
        self.assertIsNotNone(place_group['features'])

    def test_get_global_place_group(self):
        ctx = new_test_service_context()
        place_group = ctx.get_global_place_group("inside-cube", "http://localhost:9090", load_features=True)
        self.assertIsInstance(place_group, dict)
        self.assertIn("type", place_group)
        self.assertEqual("FeatureCollection", place_group["type"])
        self.assertIn("features", place_group)
        self.assertIsInstance(place_group["features"], list)
        self.assertEqual(3, len(place_group["features"]))
        self.assertIs(place_group, ctx.get_global_place_group("inside-cube", "http://localhost:9090"))
        self.assertIsNot(place_group, ctx.get_global_place_group("outside-cube", "http://localhost:9090"))

        with self.assertRaises(ServiceResourceNotFoundError) as cm:
            ctx.get_global_place_group("bibo", "http://localhost:9090")
        self.assertEqual('HTTP 404: Place group "bibo" not found', f"{cm.exception}")


class NormalizePrefixTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual('', normalize_prefix(None))
        self.assertEqual('', normalize_prefix(''))
        self.assertEqual('', normalize_prefix('/'))
        self.assertEqual('', normalize_prefix('///'))

    def test_normalizes_slashes(self):
        self.assertEqual('/xcube/api/v1', normalize_prefix('/xcube/api/v1'))
        self.assertEqual('/xcube/api/v1', normalize_prefix('xcube/api/v1'))
        self.assertEqual('/xcube/api/v1', normalize_prefix('xcube/api/v1/'))
        self.assertEqual('/xcube/api/v1', normalize_prefix('/xcube/api/v1/'))
        self.assertEqual('/xcube/api/v1', normalize_prefix('/xcube/api/v1///'))
        self.assertEqual('/xcube/api/v1', normalize_prefix('////xcube/api/v1'))

    def test_interpolates_vars(self):
        from xcube.version import version
        self.assertEqual(f'/{version}',
                         normalize_prefix('${version}'))
        self.assertEqual(f'/xcube',
                         normalize_prefix('/${name}'))
        self.assertEqual(f'/xcube/v{version}',
                         normalize_prefix('/${name}/v${version}'))
