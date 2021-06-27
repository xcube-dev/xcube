import json
import os.path
import sys
import unittest

import requests

PARENT_DIR = os.path.dirname(__file__)
SERVER_URL = 'http://127.0.0.1:5000'


class ByoaTest(unittest.TestCase):

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
            print(f'You can run the test server using:')
            print(f'$ {sys.executable} {server_path}')

    def test_service_byoa(self):
        if not self.server_running:
            return

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

        request_dict = {
            "input_configs": [
                {
                    'store_id': '@test',
                    "data_id": "DATASET-1.zarr"
                }
            ],
            "cube_config": {
            },
            "code_config": {
                "file_set": {
                    "path": user_code_filename,
                },
                "callable_ref": "processor:process_dataset",
                "callable_params": {
                    "output_var_name": "X",
                    "input_var_name_1": "A",
                    "input_var_name_2": "B",
                    "factor_1": 0.4,
                    "factor_2": 0.2,
                },
            },
            "output_config": {
                "store_id": "@test",
                "data_id": "OUTPUT.zarr",
                "replace": True,
            }
        }

        with open(user_code_path, 'rb') as stream:
            response = requests.put(
                SERVER_URL + '/generate',
                files={
                    'body': (
                        'request.json',
                        json.dumps(request_dict, indent=2),
                        'application/json'
                    ),
                    'user_code': (
                        user_code_filename,
                        open(user_code_path, 'rb'),
                        'application/octet-stream'
                    )
                }
            )
        self.assertEqual(200, response.status_code)

        response_dict = response.json()
        self.assertIsInstance(response_dict, dict)
        print(response_dict.get('output'))
        self.assertEqual(0, response_dict.get('code'))
        self.assertEqual('ok', response_dict.get('status'))
        self.assertIsInstance(response_dict.get('output'), str)
