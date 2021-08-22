import json
import os
import unittest

from xcube.core.gen2.remote.config import ServiceConfig


class ServiceConfigTest(unittest.TestCase):
    def test_normalize(self):
        json_instance = dict(
            endpoint_url='https://stage.xcube-gen.brockmann-consult.de/api/v2',
            access_token='02945ugjhklojg908ijr023jgbpij202jbv00897v0798v65472'
        )
        file_path = '_test-remote-config.json'
        with open(file_path, 'w') as fp:
            json.dump(json_instance, fp)
        try:
            service_config_1 = ServiceConfig.normalize(file_path)
            self.assertIsInstance(service_config_1, ServiceConfig)
            service_config_2 = ServiceConfig.normalize(service_config_1)
            self.assertIs(service_config_2, service_config_1)
            service_config_3 = ServiceConfig.normalize(service_config_2.to_dict())
            self.assertIsInstance(service_config_3, ServiceConfig)
        finally:
            os.remove(file_path)

        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            ServiceConfig.normalize(42)

    def test_from_file(self):
        json_instance = dict(
            endpoint_url='$_TEST_ENDPOINT_URL',
            access_token='${_TEST_ACCESS_TOKEN}'
        )
        file_path = '_test-remote-config.json'
        with open(file_path, 'w') as fp:
            json.dump(json_instance, fp)
        try:
            os.environ['_TEST_ENDPOINT_URL'] = \
                'https://stage.xcube-gen.brockmann-consult.de/api/v2'
            os.environ['_TEST_ACCESS_TOKEN'] = \
                '02945ugjhklojg908ijr023jgbpij202jbv00897v0798v65472'
            service_config = ServiceConfig.from_file(file_path)
            self.assertIsInstance(service_config, ServiceConfig)
            self.assertEqual('https://stage.xcube-gen.brockmann-consult.de/api/v2/',
                             service_config.endpoint_url)
            self.assertEqual('02945ugjhklojg908ijr023jgbpij202jbv00897v0798v65472',
                             service_config.access_token)
            self.assertEqual(None, service_config.client_id)
            self.assertEqual(None, service_config.client_secret)
        finally:
            os.remove(file_path)

    def test_from_dict(self):
        json_instance = dict(
            endpoint_url='https://stage.xcube-gen.brockmann-consult.de/api/v2',
            access_token='02945ugjhklojg908ijr023jgbpij202jbv00897v0798v65472'
        )
        service_config = ServiceConfig.from_dict(json_instance)
        self.assertIsInstance(service_config, ServiceConfig)
        self.assertEqual('https://stage.xcube-gen.brockmann-consult.de/api/v2/',
                         service_config.endpoint_url)
        self.assertEqual('02945ugjhklojg908ijr023jgbpij202jbv00897v0798v65472',
                         service_config.access_token)
        self.assertEqual(None, service_config.client_id)
        self.assertEqual(None, service_config.client_secret)

    def test_to_dict(self):
        expected_dict = dict(
            endpoint_url='https://stage.xcube-gen.brockmann-consult.de/api/v2/',
            access_token='02945ugjhklojg908ijr023jgbpij202jbv00897v0798v65472'
        )
        service_config = ServiceConfig.from_dict(expected_dict)
        actual_dict = service_config.to_dict()
        self.assertEqual(expected_dict, actual_dict)
        # smoke test JSON serialisation
        json.dumps(actual_dict, indent=2)
