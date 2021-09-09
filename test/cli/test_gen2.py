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
        self.assertEqual(
            {
                'data_id': result_zarr,
                'status': 'ok'
            },
            result_json)
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
            result_json)
        self.assertFalse(os.path.isdir(result_zarr))

    # TODO (forman): zarr writing fails because of invalid chunking
    #   Make this test work in a subsequent PR.
    #
    # def test_copy_levels_gen(self):
    #     request_file = os.path.join(os.path.dirname(__file__),
    #                                 'gen2-requests', 'copy-levels.yml')
    #     result = self.invoke_cli(['gen2',
    #                               '-o', result_file,
    #                               request_file])
    #     self.assertIsNotNone(result)
    #     result_json = self.read_result_json()
    #     self.assertEqual(
    #         {
    #             'data_id': result_levels,
    #             'status': 'ok'
    #         },
    #         result_json)
    #     self.assertTrue(os.path.isdir(result_levels))
