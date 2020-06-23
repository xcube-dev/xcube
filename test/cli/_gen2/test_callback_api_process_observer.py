import unittest

import requests_mock
from xcube.cli._gen2.callback import CallbackApiProgressObserver
from xcube.cli._gen2.request import Request
from xcube.util.progress import ProgressState


class TestCallbackApiProgressObserver(unittest.TestCase):
    REQUEST = dict(input_configs=[dict(store_id='sentinelhub',
                                       data_id='S2L2A',
                                       variable_names=['B01', 'B02', 'B03'])],
                   cube_config=dict(crs='WGS84',
                                    bbox=[12.2, 52.1, 13.9, 54.8],
                                    spatial_res=0.05,
                                    time_range=['2018-01-01', None],
                                    time_period='4D'),
                   output_config=dict(store_id='memory',
                                      data_id='CHL'),
                   callback=dict(api_uri='https://xcube-gen.test/api/v1/jobs/tomtom/iamajob/callback',
                                 access_token='dfsvdfsv'))

    def setUp(self) -> None:
        self._request = Request.from_dict(self.REQUEST)

    @requests_mock.Mocker()
    def test_observer(self, m):
        m.put('https://xcube-gen.test/api/v1/jobs/tomtom/iamajob/callback', json={})
        progress_state = ProgressState(label='test', total_work=0., super_work=10.)

        observer = CallbackApiProgressObserver(self._request)
        observer.on_begin(state_stack=[progress_state])

        with self.assertRaises(ValueError) as e:
            observer = CallbackApiProgressObserver(self._request)
            observer.on_begin(state_stack=[])

        self.assertEqual('ProgressStates must be given', str(e.exception))

        with self.assertRaises(ValueError) as e:
            self._request.callback.api_uri = None
            observer = CallbackApiProgressObserver(self._request)
            observer.on_begin(state_stack=[progress_state])

        self.assertEqual('Both, api_uri and access_token must be given.', str(e.exception))

        with self.assertRaises(ValueError) as e:
            self._request.callback.access_token = None
            observer = CallbackApiProgressObserver(self._request)
            observer.on_begin(state_stack=[progress_state])

        self.assertEqual('Both, api_uri and access_token must be given.', str(e.exception))
