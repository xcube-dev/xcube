import unittest
from typing import List, Any, Dict

import requests_mock

from test.util.test_progress import TestProgressObserver
from xcube.core.gen2 import CostEstimation
from xcube.core.gen2 import CubeGenerator
from xcube.core.gen2 import CubeInfoWithCosts
from xcube.core.gen2 import ServiceConfig
from xcube.core.gen2.remote.generator import RemoteCubeGenerator
from xcube.core.gen2.remote.response import CubeInfoWithCostsResult
from xcube.core.store import DatasetDescriptor
from xcube.util.progress import new_progress_observers


def make_result(worked, total_work,
                failed=False,
                output: List[str] = None,
                job_result: Dict[str, Any] = None):
    json = {
        "job_id": "93",
        "job_status": {
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
    if job_result is not None:
        json.update({"job_result": job_result})
    if output is not None:
        json.update({"output": output})
    return dict(json=json)


class RemoteCubeGeneratorTest(unittest.TestCase):
    ENDPOINT_URL = 'https://xcube-gen.com/api/v2/'

    REQUEST = dict(
        input_config=dict(store_id='memory',
                          data_id='S2L2A'),
        cube_config=dict(variable_names=['B01', 'B02', 'B03'],
                         crs='WGS84',
                         bbox=[12.2, 52.1, 13.9, 54.8],
                         spatial_res=0.05,
                         time_range=['2018-01-01', None],
                         time_period='4D'),
        output_config=dict(store_id='memory',
                           data_id='CHL')
    )

    def setUp(self) -> None:
        self.generator = CubeGenerator.new(
            ServiceConfig(endpoint_url=self.ENDPOINT_URL,
                          client_id='itzibitzispider',
                          client_secret='g3ergd36fd2983457fhjder'),
            verbosity=True,
            progress_period=0,
        )
        self.assertIsInstance(self.generator, RemoteCubeGenerator)

    @requests_mock.Mocker()
    def test_generate_cube_success(self, m: requests_mock.Mocker):
        m.post(f'{self.ENDPOINT_URL}oauth/token',
               json={
                   "access_token": "4ccsstkn983456jkfde",
                   "token_type": "bearer"
               })

        m.put(f'{self.ENDPOINT_URL}cubegens',
              response_list=[
                  make_result(0, 4),
              ])

        m.get(f'{self.ENDPOINT_URL}cubegens/93',
              response_list=[
                  make_result(1, 4),
                  make_result(2, 4),
                  make_result(3, 4),
                  make_result(4, 4, job_result={
                      "status": "ok",
                      "result": {
                          "data_id": "bibo.zarr"
                      }
                  }),
              ])

        observer = TestProgressObserver()
        with new_progress_observers(observer):
            self.generator.generate_cube(self.REQUEST)

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
                  make_result(0, 4),
              ])

        m.get(f'{self.ENDPOINT_URL}cubegens/93',
              response_list=[
                  make_result(1, 4),
                  make_result(2, 4,
                              failed=True,
                              output=['1.that', '2.was', '3.bad']),
              ])

        observer = TestProgressObserver()
        with new_progress_observers(observer):
            cube_result = self.generator.generate_cube(self.REQUEST)
            self.assertEqual('error', cube_result.status)
            self.assertEqual(['1.that', '2.was', '3.bad'], cube_result.output)

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
                   "status": "ok",
                   "result": {
                       "dataset_descriptor": {
                           "data_id": "CHL",
                           "data_type": "dataset",
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
                   }
               })

        result = self.generator.get_cube_info(self.REQUEST)
        self.assertIsInstance(result, CubeInfoWithCostsResult)
        cube_info = result.result
        self.assertIsInstance(cube_info, CubeInfoWithCosts)
        self.assertIsInstance(cube_info.dataset_descriptor, DatasetDescriptor)
        self.assertIsInstance(cube_info.size_estimation, dict)
        self.assertIsInstance(cube_info.cost_estimation, CostEstimation)
        self.assertEqual(3782, cube_info.cost_estimation.required)
        self.assertEqual(234979, cube_info.cost_estimation.available)
        self.assertEqual(10000, cube_info.cost_estimation.limit)
