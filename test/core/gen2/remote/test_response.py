import unittest

from xcube.core.gen2.remote.response import CostEstimation
from xcube.core.gen2.remote.response import CubeInfoWithCosts
from xcube.core.gen2.remote.response import CubeInfoWithCostsResult
from xcube.core.gen2.response import GenericCubeGeneratorResult
from xcube.core.store import DatasetDescriptor


class CubeInfoWithCostsResultTest(unittest.TestCase):
    def test_serialisation(self):
        result = CubeInfoWithCostsResult(
            status='ok',
            message='Success!',
            result=CubeInfoWithCosts(
                dataset_descriptor=DatasetDescriptor(data_id='bibo.zarr'),
                size_estimation={},
                cost_estimation=CostEstimation(required=10,
                                               available=20,
                                               limit=100)
            )
        )

        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(
            {
                'status': 'ok',
                'message': 'Success!',
                'result': {
                    'dataset_descriptor': {
                        'data_id': 'bibo.zarr',
                        'data_type': 'dataset'
                    },
                    'size_estimation': {},
                    'cost_estimation': {
                        'available': 20,
                        'limit': 100,
                        'required': 10
                    },
                },
            },
            result_dict)

        result2 = CubeInfoWithCostsResult.from_dict(result_dict)
        self.assertIsInstance(result2, GenericCubeGeneratorResult)
        self.assertIsInstance(result2.result, CubeInfoWithCosts)
