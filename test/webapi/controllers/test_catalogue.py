import unittest

from xcube.webapi.context import ServiceContext
from xcube.webapi.controllers.catalogue import get_datasets, get_color_bars
from xcube.webapi.errors import ServiceBadRequestError
from ..helpers import new_test_service_context


class CatalogueControllerTest(unittest.TestCase):
    def test_datasets(self):
        ctx = new_test_service_context()

        response = get_datasets(ctx)
        self.assertIsInstance(response, dict)
        self.assertIn("datasets", response)
        self.assertIsInstance(response["datasets"], list)
        self.assertEqual(2, len(response["datasets"]))
        dataset = response["datasets"][0]
        self.assertIsInstance(dataset, dict)
        self.assertIn("id", dataset)
        self.assertIn("title", dataset)
        self.assertNotIn("variables", dataset)
        self.assertNotIn("dimensions", dataset)

    def test_dataset_with_details(self):
        ctx = new_test_service_context()

        response = get_datasets(ctx, details=True, base_url="http://test")
        self.assertIsInstance(response, dict)
        self.assertIn("datasets", response)
        self.assertIsInstance(response["datasets"], list)
        self.assertEqual(2, len(response["datasets"]))
        dataset = response["datasets"][0]
        self.assertIsInstance(dataset, dict)
        self.assertIn("id", dataset)
        self.assertIn("title", dataset)
        self.assertIn("variables", dataset)
        self.assertIn("dimensions", dataset)

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
