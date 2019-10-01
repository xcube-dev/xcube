import os.path
import unittest

import xarray as xr

from xcube.webapi.errors import ServiceResourceNotFoundError
from .helpers import new_test_service_context


class ServiceContextTest(unittest.TestCase):
    def test_get_dataset_and_variable(self):
        ctx = new_test_service_context()

        ds = ctx.get_dataset('demo')
        self.assertIsInstance(ds, xr.Dataset)

        with self.assertRaises(ServiceResourceNotFoundError) as cm:
            ctx.get_dataset('demox')
        self.assertEqual(404, cm.exception.status_code)
        self.assertEqual('Dataset "demox" not found', cm.exception.reason)

        with self.assertRaises(ServiceResourceNotFoundError) as cm:
            ctx.get_dataset('demo', expected_var_names=['conc_ys'])
        self.assertEqual(404, cm.exception.status_code)
        self.assertEqual('Variable "conc_ys" not found in dataset "demo"', cm.exception.reason)

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
        self.assertEqual(('plasma', 0., 24.), cm)
        cm = ctx.get_color_mapping('demo', 'conc_tsm')
        self.assertEqual(('PuBuGn', 0., 100.), cm)
        cm = ctx.get_color_mapping('demo', 'kd489')
        self.assertEqual(('jet', 0., 6.), cm)
        cm = ctx.get_color_mapping('demo', '_')
        self.assertEqual(('jet', 0., 1.), cm)

    def test_get_global_place_groups(self):
        ctx = new_test_service_context()
        place_groups = ctx.get_global_place_groups(load_features=False)
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

        place_groups = ctx.get_global_place_groups(load_features=True)
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
        place_group = ctx.get_global_place_group("inside-cube", load_features=True)
        self.assertIsInstance(place_group, dict)
        self.assertIn("type", place_group)
        self.assertEqual("FeatureCollection", place_group["type"])
        self.assertIn("features", place_group)
        self.assertIsInstance(place_group["features"], list)
        self.assertEqual(3, len(place_group["features"]))
        self.assertIs(place_group, ctx.get_global_place_group(place_group_id="inside-cube"))
        self.assertIsNot(place_group, ctx.get_global_place_group(place_group_id="outside-cube"))

        with self.assertRaises(ServiceResourceNotFoundError) as cm:
            ctx.get_global_place_group("bibo")
        self.assertEqual('HTTP 404: Place group "bibo" not found', f"{cm.exception}")
