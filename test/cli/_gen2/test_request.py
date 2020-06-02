import unittest

from xcube.cli._gen2.request import Request


class RequestTest(unittest.TestCase):

    def test_to_and_from_dict(self):
        request_dict = dict(input_configs=[dict(data_store_id='mem', dataset_id='x')],
                            cube_config=dict(),
                            code_config=dict(),
                            output_config=dict(data_store_id='mem', dataset_id='y'))
        request = Request.from_dict(request_dict)
        request_dict_2 = request.to_dict()
        return self.assertEqual(request_dict, request_dict_2)
