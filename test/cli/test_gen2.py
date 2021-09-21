import json
import os
import os.path

from test.cli.helpers import CliTest
from xcube.core.dsio import rimraf

result_file = 'result.json'
result_zarr = 'out.zarr'
result_levels = 'out.levels'


class Gen2CliTest(CliTest):

    def setUp(self) -> None:
        rimraf(result_file, result_zarr, result_levels)

    def tearDown(self) -> None:
        rimraf(result_file, result_zarr, result_levels)

    def read_result_json(self):
        self.assertTrue(os.path.exists(result_file))
        with open(result_file) as fp:
            result_json = json.load(fp)
        self.assertIsInstance(result_json, dict)
        return result_json

    def test_help(self):
        result = self.invoke_cli(['gen2', '--help'])
        self.assertEqual(0, result.exit_code)

    def test_copy_zarr_gen(self):
        request_file = os.path.join(os.path.dirname(__file__),
                                    'gen2-requests', 'copy-zarr.yml')
        result = self.invoke_cli(['gen2',
                                  '-o', result_file,
                                  request_file])
        self.assertIsNotNone(result)
        result_json = self.read_result_json()
        self.assertEqual('ok', result_json.get('status'))
        self.assertEqual(201, result_json.get('status_code'))
        self.assertIsInstance(result_json.get('message'), str)
        self.assertIn('Cube generated successfully after ',
                      result_json.get('message'))
        result = result_json.get('result')
        self.assertIsInstance(result, dict)
        self.assertEqual(result_zarr, result.get('data_id'))
        self.assertTrue(os.path.isdir(result_zarr))

    def test_copy_zarr_info(self):
        request_file = os.path.join(os.path.dirname(__file__),
                                    'gen2-requests', 'copy-zarr.yml')
        result = self.invoke_cli(['gen2',
                                  '--info',
                                  '-o', result_file,
                                  request_file])
        self.assertIsNotNone(result)
        result_json = self.read_result_json()

        self.assertEqual('ok', result_json.get('status'))
        self.assertEqual(200, result_json.get('status_code'))
        result = result_json.get('result')
        self.assertIsInstance(result, dict)

        self.assertEqual(
            {
                'dataset_descriptor': {
                    'data_id': result_zarr,
                    'data_type': 'dataset',
                    'dims': {'lat': 1000, 'lon': 2000, 'time': 15},
                    'spatial_res': 0.0024999999999977263,
                    'time_period': '1D',
                    'time_range': ['2017-01-16', '2017-01-30'],
                    'bbox': [0.0, 50.0, 5.0, 52.5],
                    'crs': 'WGS84',
                    'data_vars': {
                        'c2rcc_flags': {
                            'dims': ['time', 'lat', 'lon'],
                            'dtype': 'float32',
                            'name': 'c2rcc_flags'
                        },
                        'conc_chl': {
                            'dims': ['time', 'lat', 'lon'],
                            'dtype': 'float32',
                            'name': 'conc_chl'
                        },
                        'conc_tsm': {
                            'dims': ['time', 'lat', 'lon'],
                            'dtype': 'float32',
                            'name': 'conc_tsm'
                        },
                        'kd489': {
                            'dims': ['time', 'lat', 'lon'],
                            'dtype': 'float32',
                            'name': 'kd489'
                        },
                        'quality_flags': {
                            'dims': ['time', 'lat', 'lon'],
                            'dtype': 'float32',
                            'name': 'quality_flags'
                        }
                    },
                },
                'size_estimation': {
                    'image_size': [2000, 1000],
                    'num_bytes': 600000000,
                    'num_requests': 75,
                    'num_tiles': [1, 1],
                    'num_variables': 5,
                    'tile_size': [2000, 1000]
                }
            },
            result)
        self.assertFalse(os.path.isdir(result_zarr))

    def test_copy_levels_gen(self):
        request_file = os.path.join(os.path.dirname(__file__),
                                    'gen2-requests', 'copy-levels.yml')
        result = self.invoke_cli(['gen2',
                                  '-o', result_file,
                                  request_file])
        print(result.output)
        self.assertIsNotNone(result)
        result_json = self.read_result_json()
        self.assertEqual('ok', result_json.get('status'))
        self.assertEqual(201, result_json.get('status_code'))
        self.assertEqual({'data_id': 'out.levels'}, result_json.get('result'))
        self.assertTrue(os.path.isdir(result_levels))

    def test_internal_error(self):
        request_file = os.path.join(os.path.dirname(__file__),
                                    'gen2-requests', 'internal-error.yml')
        result = self.invoke_cli(['gen2',
                                  '-o', result_file,
                                  request_file])
        print(result.output)
        self.assertIsNotNone(result)
        result_json = self.read_result_json()
        self.assertEqual('error', result_json.get('status'))
        self.assertEqual(500, result_json.get('status_code'))
        self.assertEqual('inverse_fine_structure_constant must be 137'
                         ' or running in wrong universe',
                         result_json.get('message'))
        self.assertNotIn('result', result_json)
