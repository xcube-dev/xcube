import unittest

import requests_mock

from test.cli._gen2.config import XCUBE_API_CFG
from xcube.cli._gen2.callback import CallbackApiProgressObserver
from xcube.cli._gen2.request import Request
from xcube.util.progress import ProgressState


class TestCallbackApiProgressObserver(unittest.TestCase):
    @requests_mock.Mocker()
    def test_observer(self, m):
        m.put('https://xcube-gen.test/api/v1/', json={})
        progress_state = ProgressState(label='test', total_work=0., super_work=10.)

        request = Request.from_dict(XCUBE_API_CFG)

        observer = CallbackApiProgressObserver(request)
        observer.on_begin(state_stack=[progress_state])

        with self.assertRaises(ValueError) as e:
            observer = CallbackApiProgressObserver(request)
            observer.on_begin(state_stack=[])

        self.assertEqual('ProgressStates must be given', str(e.exception))

        with self.assertRaises(ValueError) as e:
            request.callback.api_uri = None
            observer = CallbackApiProgressObserver(request)
            observer.on_begin(state_stack=[progress_state])

        self.assertEqual('Both, api_uri and access_token must be given.', str(e.exception))

        with self.assertRaises(ValueError) as e:
            request.callback.access_token = None
            observer = CallbackApiProgressObserver(request)
            observer.on_begin(state_stack=[progress_state])

        self.assertEqual('Both, api_uri and access_token must be given.', str(e.exception))
