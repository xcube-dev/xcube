import json
import os.path
import sys
import unittest

import requests

from xcube.core.gen2 import CubeGeneratorError
from xcube.core.gen2 import ServiceConfig
from xcube.core.gen2.remote.generator import RemoteCubeGenerator

PARENT_DIR = os.path.dirname(__file__)
SERVER_URL = 'http://127.0.0.1:5000'


class ServiceTest(unittest.TestCase):
    """
    This test uses a real cube generator service
    and asserts that the RemoteCubeGenerator works as expected.

    It requires you to first start "test/core/gen2/remote/server.py"
    which is a working processing cube generator server compatible
    with the expected xcube Generator REST API.

    This test sets up a generator request with a use-code configuration
    that will cause the generator to
    1. open a dataset "DATASET-1.zarr" in store "@test"
    2. invoke function "process_dataset()" in module "processor" from
       "test/core/byoa/test_data/user_code.zip".
    3. write a dataset "OUTPUT.zarr" in store "@test"
    """

    server_running = False

    @classmethod
    def setUpClass(cls):
        try:
            response = requests.get(SERVER_URL + '/status')
            cls.server_running = response.ok
        except requests.exceptions.ConnectionError:
            cls.server_running = False
        if not cls.server_running:
            server_path = os.path.join(PARENT_DIR, "server.py")
            print(f'Tests in {__file__} ignored,')
            print(f'test server at {SERVER_URL} is not running.')
            print(f'You can start the test server using:')
            print(f'$ {sys.executable} {server_path}')

    def test_service_byoa_inline_code(self):
        """
        Assert the service can run BYOA with a inline code.

        This test sets up a generator request with an
        inline user-code configuration
        that will cause the generator to

        1. open a dataset "DATASET-1.zarr" in store "@test"
        2. invoke function "process_dataset()" in module "processor" from
           "test/core/byoa/test_data/user_code.zip".
        3. write a dataset "OUTPUT.zarr" in store "@test"

        """
        self._test_service_byoa(inline_code=True)

    def test_service_byoa_code_package(self):
        """
        Assert the service can run BYOA with a code package.

        This test sets up a generator request with an
        packaged user-code configuration
        that will cause the generator to

        1. open a dataset "DATASET-1.zarr" in store "@test"
        2. invoke function "process_dataset()" in module "processor" from
           "test/core/byoa/test_data/user_code.zip".
        3. write a dataset "OUTPUT.zarr" in store "@test"

        """
        self._test_service_byoa(inline_code=False)

    def _test_service_byoa(self, inline_code: bool):
        if not self.server_running:
            return

        callable_params = {
            "output_var_name": "X",
            "input_var_name_1": "A",
            "input_var_name_2": "B",
            "factor_1": 0.4,
            "factor_2": 0.2,
        }

        if inline_code:
            inline_code = (
                "def process_dataset(dataset,\n"
                "                    output_var_name,\n"
                "                    input_var_name_1,\n"
                "                    input_var_name_2,\n"
                "                    factor_1, factor_2):\n"
                "    return dataset.assign(**{\n"
                "        output_var_name: ("
                "            factor_1 * dataset[input_var_name_1] + "
                "            factor_2 * dataset[input_var_name_2]"
                "        )\n"
                "    })"
            )
            code_config = {
                "inline_code": inline_code,
                "callable_ref": "user_code:process_dataset",
                "callable_params": callable_params,
            }
        else:
            user_code_filename = 'user_code.zip'

            user_code_path = os.path.normpath(
                os.path.join(PARENT_DIR,
                             '..',
                             '..',
                             'byoa',
                             'test_data',
                             user_code_filename)
            )
            self.assertTrue(os.path.exists(user_code_path), msg=user_code_path)

            code_config = {
                "file_set": {
                    "path": user_code_path,
                },
                "callable_ref": "processor:process_dataset",
                "callable_params": callable_params,
            }

        request_dict = {
            "input_configs": [
                {
                    'store_id': '@test',
                    "data_id": "DATASET-1.zarr"
                }
            ],
            "cube_config": {
            },
            "code_config": code_config,
            "output_config": {
                "store_id": "@test",
                "data_id": "OUTPUT.zarr",
                "replace": True,
            }
        }

        service = RemoteCubeGenerator(
            service_config=ServiceConfig(endpoint_url=SERVER_URL)
        )

        try:
            result = service.get_cube_info(request_dict)
            result_json = result.to_dict()
            print('Cube info result:\n',
                  json.dumps(result_json, indent=2),
                  flush=True)
            self.assertEqual(
                {
                    'status': 'ok',
                    'status_code': 200,
                    'result': {
                        'dataset_descriptor': {
                            'bbox': [-180.0, -90.0, -144.0, -72.0],
                            'crs': 'WGS84',
                            'data_id': 'OUTPUT.zarr',
                            'data_type': 'dataset',
                            'data_vars': {
                                'A': {'dims': ['time',
                                               'lat',
                                               'lon'],
                                      'dtype': 'float32',
                                      'name': 'A'},
                                'B': {'dims': ['time',
                                               'lat',
                                               'lon'],
                                      'dtype': 'float32',
                                      'name': 'B'}
                            },
                            'dims': {'lat': 18, 'lon': 36, 'time': 6},
                            'spatial_res': 1.0,
                            'time_period': '1D',
                            'time_range': ['2010-01-01', '2010-01-06']
                        },
                        'size_estimation': {
                            'image_size': [36, 18],
                            'num_bytes': 31104,
                            'num_requests': 12,
                            'num_tiles': [1, 1],
                            'num_variables': 2,
                            'tile_size': [36, 18]
                        }
                    },
                },
                result_json
            )
        except CubeGeneratorError as e:
            print('Status code:\n', e.status_code, flush=True)
            print('Remote output:\n', e.remote_output, flush=True)
            print('Remote traceback:\n', e.remote_traceback, flush=True)
            self.fail(f'CubeGeneratorError: get_cube_info() failed')

        try:
            result = service.generate_cube(request_dict)
            result_json = result.to_dict()
            print('Cube generator result:',
                  json.dumps(result_json, indent=2),
                  flush=True)
            self.assertEqual('ok',
                             result_json.get('status'))
            self.assertEqual(201,
                             result_json.get('status_code'))
            self.assertEqual({'data_id': 'OUTPUT.zarr'},
                             result_json.get('result'))
        except CubeGeneratorError as e:
            print('Status code:\n', e.status_code, flush=True)
            print('Remote output:\n', e.remote_output, flush=True)
            print('Remote traceback:\n', e.remote_traceback, flush=True)
            self.fail(f'CubeGeneratorError: generate_cube() failed')

    def test_service_simple_copy(self):
        """
        Assert the service can copy.

        1. opens a dataset "DATASET-1.zarr" in store "@test"
        2. writes this dataset "OUTPUT.zarr" in store "@test"
        """
        if not self.server_running:
            return

        generator = RemoteCubeGenerator(
            service_config=ServiceConfig(endpoint_url=SERVER_URL)
        )

        request_dict = {
            "input_configs": [
                {
                    'store_id': '@test',
                    "data_id": "DATASET-1.zarr"
                }
            ],
            "output_config": {
                "store_id": "@test",
                "data_id": "OUTPUT.zarr",
                "replace": True,
            }
        }

        result = generator.generate_cube(request_dict)

        result_dict = result.to_dict()
        self.assertEqual('ok', result_dict.get('status'))
        self.assertEqual(201, result_dict.get('status_code'))
        self.assertEqual({'data_id': 'OUTPUT.zarr'}, result_dict.get('result'))

    def test_service_empty_cube(self):
        """
        Assert the service returns an empty-cube warning.
        """
        if not self.server_running:
            return

        generator = RemoteCubeGenerator(
            service_config=ServiceConfig(endpoint_url=SERVER_URL)
        )

        request_dict = {
            "input_configs": [
                {
                    'store_id': '@test',
                    "data_id": "DATASET-1.zarr"
                }
            ],
            "cube_config": {
                "time_range": ["1981-01-01", "1981-02-01"],
            },
            "output_config": {
                "store_id": "@test",
                "data_id": "OUTPUT.zarr",
                "replace": True,
            }
        }

        result = generator.generate_cube(request_dict)

        result_dict = result.to_dict()
        self.assertEqual('warning', result_dict.get('status'))
        self.assertEqual(422, result_dict.get('status_code'))
        self.assertEqual(None, result_dict.get('result'))
        self.assertIn('An empty cube has been generated ',
                      result_dict.get('message', ''))
        self.assertIn('No data has been written at all.',
                      result_dict.get('message', ''))
        self.assertEqual(None, result_dict.get('output'))

    def test_service_invalid_request(self):
        """
        Assert the service recognizes invalid requests.
        """
        if not self.server_running:
            return

        generator = RemoteCubeGenerator(
            service_config=ServiceConfig(endpoint_url=SERVER_URL)
        )

        request_dict = {
            "input_configs": [
                {
                    'store_id': '@test',
                    "data_id": "DATASET-8.zarr"
                }
            ],
            "output_config": {
                "store_id": "@test",
                "data_id": "OUTPUT.zarr",
                "replace": True,
            }
        }

        result = generator.generate_cube(request_dict)

        result_dict = result.to_dict()
        self.assertEqual('error', result_dict.get('status'))
        self.assertEqual(400, result_dict.get('status_code'))
        self.assertEqual(None, result_dict.get('result'))
        self.assertEqual('Data resource "DATASET-8.zarr"'
                         ' does not exist in store',
                         result_dict.get('message'))
        self.assertEqual(None, result_dict.get('output'))

    def test_service_internal_error(self):
        """
        Assert the service handles internal errors.
        """

        if not self.server_running:
            return

        generator = RemoteCubeGenerator(
            service_config=ServiceConfig(endpoint_url=SERVER_URL)
        )

        request_dict = {
            "input_configs": [
                {
                    'store_id': '@test',
                    "data_id": "DATASET-1.zarr"
                }
            ],
            "cube_config": {
                "metadata": {
                    # This raises a ValueError in "xcube gen2"
                    'inverse_fine_structure_constant': 138
                }
            },
            "output_config": {
                "store_id": "@test",
                "data_id": "OUTPUT.zarr",
                "replace": True,
            }
        }

        result = generator.generate_cube(request_dict)

        result_dict = result.to_dict()
        self.assertEqual('error', result_dict.get('status'))
        self.assertEqual(400, result_dict.get('status_code'))
        self.assertEqual(None, result_dict.get('result'))
        self.assertEqual('inverse_fine_structure_constant must be 137'
                         ' or running in wrong universe',
                         result_dict.get('message'))
        self.assertEqual(None, result_dict.get('output'))

        # self.assertEqual(('inverse_fine_structure_constant must be 137'
        #                   ' or running in wrong universe',),
        #                  cm.exception.args)
