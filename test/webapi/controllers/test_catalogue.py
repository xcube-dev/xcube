import unittest
from typing import Any

from test.webapi.helpers import new_test_service_context
from xcube.webapi.context import ServiceContext
from xcube.webapi.controllers.catalogue import get_datasets, get_color_bars
from xcube.webapi.errors import ServiceBadRequestError


class CatalogueControllerTest(unittest.TestCase):
    def test_datasets(self):
        ctx = new_test_service_context()
        response = get_datasets(ctx)
        datasets = self._assert_datasets(response, 2)
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
        datasets = self._assert_datasets(response, 2)

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
        datasets = self._assert_datasets(response, 1)
        dataset = datasets[0]
        self.assertIsInstance(dataset, dict)
        self.assertEqual({'varNames': ['conc_chl', 'conc_tsm', 'kd489'],
                          'normRanges': [(0.0, 24.0), (0.0, 100.0), (0.0, 6.0)]},
                         dataset.get("rgbSchema"))
        response = get_datasets(ctx, details=True, client='ol4', base_url="http://test")
        datasets = self._assert_datasets(response, 1)
        dataset = datasets[0]
        self.assertIsInstance(dataset, dict)
        self.assertEqual({'varNames': ['conc_chl', 'conc_tsm', 'kd489'],
                          'normRanges': [(0.0, 24.0), (0.0, 100.0), (0.0, 6.0)],
                          'tileSourceOptions': {
                              'url': 'http://test/datasets/demo-rgb/vars/rgb/tiles/{z}/{x}/{y}.png',
                              'projection': 'EPSG:4326',
                              'tileGrid': {'extent': [0, 50, 5, 52.5],
                                           'origin': [0, 52.5],
                                           'resolutions': [0.01, 0.005, 0.0025],
                                           'sizes': [[2, 1], [4, 2], [8, 4]],
                                           'tileSize': [250, 250]}
                          }},
                         dataset.get("rgbSchema"))

    def test_dataset_with_point(self):
        ctx = new_test_service_context()

        response = get_datasets(ctx, point=(1.7, 51.2), base_url="http://test")
        datasets = self._assert_datasets(response, 2)
        dataset = datasets[0]
        self.assertIsInstance(dataset, dict)
        self.assertIn("id", dataset)
        self.assertIn("title", dataset)
        self.assertNotIn("variables", dataset)
        self.assertNotIn("dimensions", dataset)

        response = get_datasets(ctx, point=(1.7, 58.0), base_url="http://test")
        self._assert_datasets(response, 0)

    def test_dataset_with_point_and_details(self):
        ctx = new_test_service_context()

        response = get_datasets(ctx, point=(1.7, 51.2), details=True, base_url="http://test")
        datasets = self._assert_datasets(response, 2)
        dataset = datasets[0]
        self.assertIsInstance(dataset, dict)
        self.assertIn("id", dataset)
        self.assertIn("title", dataset)
        self.assertIn("variables", dataset)
        self.assertIn("dimensions", dataset)

        response = get_datasets(ctx, point=(1.7, 58.0), details=True, base_url="http://test")
        self._assert_datasets(response, 0)

    def test_get_colorbars(self):
        ctx = ServiceContext()

        response = get_color_bars(ctx, 'application/json')
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 40)
        self.assertEqual('[\n  [\n    "Perceptually Uniform Sequenti', response[0:40])

        response = get_color_bars(ctx, 'text/html')
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 40)
        self.assertEqual('<!DOCTYPE html>\n<html lang="en">\n<head><', response[0:40])

        with self.assertRaises(ServiceBadRequestError) as cm:
            get_color_bars(ctx, 'text/xml')
        self.assertEqual(400, cm.exception.status_code)
        self.assertEqual("Format 'text/xml' not supported for color bars", cm.exception.reason)

    def _assert_datasets(self, response: Any, expected_count: int):
        self.assertIsInstance(response, dict)
        datasets = response.get("datasets")
        self.assertIsInstance(datasets, list)
        self.assertEqual(expected_count, len(datasets))
        return datasets
