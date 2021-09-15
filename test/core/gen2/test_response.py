import unittest

from xcube.core.gen2.response import CubeGeneratorResult
from xcube.core.gen2.response import CubeInfo
from xcube.core.gen2.response import CubeInfoResult
from xcube.core.gen2.response import CubeReference
from xcube.core.store import DatasetDescriptor


class CubeGeneratorResultTest(unittest.TestCase):
    def test_serialisation(self):
        result = CubeGeneratorResult(status='ok',
                                     message='Success!',
                                     result=CubeReference(data_id='bibo.zarr'))

        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(
            {
                'status': 'ok',
                'message': 'Success!',
                'result': {'data_id': 'bibo.zarr'},
            },
            result_dict)

        result2 = CubeGeneratorResult.from_dict(result_dict)
        self.assertIsInstance(result2, CubeGeneratorResult)
        self.assertIsInstance(result2.result, CubeReference)
        self.assertEqual('bibo.zarr', result2.result.data_id)


class CubeInfoResultTest(unittest.TestCase):
    def test_serialisation(self):
        result = CubeInfoResult(
            status='ok',
            message='Success!',
            result=CubeInfo(
                dataset_descriptor=DatasetDescriptor(data_id='bibo.zarr'),
                size_estimation={}
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
                    'size_estimation': {}
                },
            },
            result_dict)

        result2 = CubeInfoResult.from_dict(result_dict)
        self.assertIsInstance(result2, CubeInfoResult)
