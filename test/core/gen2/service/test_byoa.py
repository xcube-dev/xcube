import json
import os.path
import sys
import unittest

import requests

from xcube.core.gen2 import CubeGeneratorRequest, CubeGeneratorError
from xcube.core.gen2.service import RemoteCubeGenerator
from xcube.core.gen2.service import ServiceConfig

PARENT_DIR = os.path.dirname(__file__)
SERVER_URL = 'http://127.0.0.1:5000'


class ByoaTest(unittest.TestCase):
    """
    This test demonstrates the BYOA feature within the
    Generator service.
    It requires you to first start "test/core/gen2/service/server.py"
    which is a working processing service compatible with the
    actual xcube Generator REST API.

    This test sets up a generator request with a use-code configuration
    that will cause the generator to
    1. open a dataset "DATASET-1.zarr" in store "@test"
    2. invoke function "process_dataset()" in module "processor" from
       "test/core/byoa/test_data/user_code.zip".
    3. write a dataset "OUTPUT.zarr" in store "@test"
    """

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

    def test_service_inline_code(self):
        self._test_service_byoa(inline_code=True)

    def test_service_code_package(self):
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
            CubeGeneratorRequest.from_dict(request_dict),
            ServiceConfig(endpoint_url=SERVER_URL)
        )

        try:
            cube_info = service.get_cube_info()
            print('Cube Info:\n', json.dumps(cube_info.to_dict(), indent=2),
                  flush=True)
        except CubeGeneratorError as e:
            print('Remote output:\n', e.remote_output, flush=True)
            print('Remote traceback:\n', e.remote_traceback, flush=True)
            self.fail(f'CubeGeneratorError: get_cube_info() failed')

        try:
            cube_id = service.generate_cube()
            print('Cube ID:', cube_id,
                  flush=True)
        except CubeGeneratorError as e:
            print('Remote output:\n', e.remote_output, flush=True)
            print('Remote traceback:\n', e.remote_traceback, flush=True)
            self.fail(f'CubeGeneratorError: generate_cube() failed')

        self.assertEqual('OUTPUT.zarr', cube_id)
        # TODO: verify OUTPUT.zarr contains variable "X" with expected values
        #   Current problem: we have no clue where it is :)
