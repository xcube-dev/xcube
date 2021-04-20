import unittest
from typing import List

import requests_mock

from test.util.test_progress import TestProgressObserver
from xcube.core.gen2 import CubeGeneratorError
from xcube.core.gen2.request import CubeGeneratorRequest
from xcube.core.gen2.service import CubeGeneratorService
from xcube.core.gen2.service import ServiceConfig
from xcube.core.gen2.service.response import CubeInfoWithCosts
from xcube.core.gen2.service.response import CostEstimation
from xcube.core.store import DatasetDescriptor
from xcube.util.progress import new_progress_observers


def result(worked, total_work, failed=False, output: List[str] = None):
    json = {
        "cubegen_id": "93",
        "status": {
            "failed": 1 if failed else None,
            "succeeded": 1 if worked == total_work else None,
            "active": 1 if worked != total_work else None,
        },
        "progress": [
            {
                "sender": "ignored",
                "state": {
                    "progress": worked / total_work,
                    "worked": worked,
                    "total_work": total_work,
                }
            },
        ],
    }
    if output:
        json.update(output=output)
    return dict(json=json)


class CubeGeneratorServiceTest(unittest.TestCase):
    ENDPOINT_URL = 'https://xcube-gen.com/api/v2/'

    CUBE_GEN_CONFIG = dict(input_config=dict(store_id='memory',
                                             data_id='S2L2A'),
                           cube_config=dict(variable_names=['B01', 'B02', 'B03'],
                                            crs='WGS84',
                                            bbox=[12.2, 52.1, 13.9, 54.8],
                                            spatial_res=0.05,
                                            time_range=['2018-01-01', None],
                                            time_period='4D'),
                           output_config=dict(store_id='memory',
                                              data_id='CHL'))

    def setUp(self) -> None:
        self.service = CubeGeneratorService(CubeGeneratorRequest.from_dict(self.CUBE_GEN_CONFIG),
                                            ServiceConfig(endpoint_url=self.ENDPOINT_URL,
                                                          client_id='itzibitzispider',
                                                          client_secret='g3ergd36fd2983457fhjder'),
                                            progress_period=0,
                                            verbosity=True)

    @requests_mock.Mocker()
    def test_generate_cube_success(self, m: requests_mock.Mocker):
        m.post(f'{self.ENDPOINT_URL}oauth/token',
               json={
                   "access_token": "4ccsstkn983456jkfde",
                   "token_type": "bearer"
               })

        m.put(f'{self.ENDPOINT_URL}cubegens',
              response_list=[
                  result(0, 4),
              ])

        m.get(f'{self.ENDPOINT_URL}cubegens/93',
              response_list=[
                  result(1, 4),
                  result(2, 4),
                  result(3, 4),
                  result(4, 4),
              ])

        observer = TestProgressObserver()
        with new_progress_observers(observer):
            self.service.generate_cube()

        self.assertEqual(
            [
                ('begin', [('Generating cube', 0.0, False)]),
                ('update', [('Generating cube', 0.25, False)]),
                ('update', [('Generating cube', 0.5, False)]),
                ('update', [('Generating cube', 0.75, False)]),
                ('end', [('Generating cube', 0.75, True)])
            ],
            observer.calls)

    @requests_mock.Mocker()
    def test_generate_cube_failure(self, m: requests_mock.Mocker):
        m.post(f'{self.ENDPOINT_URL}oauth/token',
               json={
                   "access_token": "4ccsstkn983456jkfde",
                   "token_type": "bearer"
               })

        m.put(f'{self.ENDPOINT_URL}cubegens',
              response_list=[
                  result(0, 4),
              ])

        m.get(f'{self.ENDPOINT_URL}cubegens/93',
              response_list=[
                  result(1, 4),
                  result(2, 4, failed=True, output=['1.that', '2.was', '3.bad']),
              ])

        observer = TestProgressObserver()
        with new_progress_observers(observer):
            with self.assertRaises(CubeGeneratorError) as cm:
                self.service.generate_cube()
            self.assertEqual('Cube generation failed', f'{cm.exception}')
            self.assertEqual(['1.that', '2.was', '3.bad'], cm.exception.remote_output)

        print(observer.calls)
        self.assertEqual(
            [
                ('begin', [('Generating cube', 0.0, False)]),
                ('update', [('Generating cube', 0.25, False)]),
                ('end', [('Generating cube', 0.25, True)])
            ],
            observer.calls)

    @requests_mock.Mocker()
    def test_get_cube_info(self, m: requests_mock.Mocker):
        m.post(f'{self.ENDPOINT_URL}oauth/token',
               json={
                   "access_token": "4ccsstkn983456jkfde",
                   "token_type": "bearer"
               })

        m.post(f'{self.ENDPOINT_URL}cubegens/info',
               json={
                   "dataset_descriptor": {
                       "type_specifier": "dataset",
                       "data_id": "CHL",
                       "crs": "WGS84",
                       "bbox": [12.2, 52.1, 13.9, 54.8],
                       "time_range": ["2018-01-01", "2010-01-06"],
                       "time_period": "4D",
                       "data_vars": {
                           "B01": {
                               "name": "B01",
                               "dtype": "float32",
                               "dims": ["time", "lat", "lon"],
                           },
                           "B02": {
                               "name": "B02",
                               "dtype": "float32",
                               "dims": [
                                   "time",
                                   "lat",
                                   "lon"
                               ],
                           },
                           "B03": {
                               "name": "B03",
                               "dtype": "float32",
                               "dims": ["time", "lat", "lon"],
                           }
                       },
                       "spatial_res": 0.05,
                       "dims": {"time": 0, "lat": 54, "lon": 34}
                   },
                   "size_estimation": {
                       "image_size": [34, 54],
                       "tile_size": [34, 54],
                       "num_variables": 3,
                       "num_tiles": [1, 1],
                       "num_requests": 0,
                       "num_bytes": 0
                   },
                   "cost_estimation": {
                       'required': 3782,
                       'available': 234979,
                       'limit': 10000
                   }
               })

        cube_info = self.service.get_cube_info()
        self.assertIsInstance(cube_info, CubeInfoWithCosts)
        self.assertIsInstance(cube_info.dataset_descriptor, DatasetDescriptor)
        self.assertIsInstance(cube_info.size_estimation, dict)
        self.assertIsInstance(cube_info.cost_estimation, CostEstimation)
        self.assertEqual(3782, cube_info.cost_estimation.required)
        self.assertEqual(234979, cube_info.cost_estimation.available)
        self.assertEqual(10000, cube_info.cost_estimation.limit)
