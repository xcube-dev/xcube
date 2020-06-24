import unittest
from time import sleep
from typing import Sequence

import requests_mock
from xcube.cli._gen2.progress_observer import CallbackApiProgressObserver, CallbackTerminalProgressObserver, \
    ThreadedProgressObserver
from xcube.cli._gen2.request import Request
from xcube.util.progress import ProgressState, observe_progress


class TestThreadedProgressObserver(unittest.TestCase):
    class _TestThreaded(ThreadedProgressObserver):
        def __init__(self, minimum=0, dt=0):
            super().__init__(minimum=minimum, dt=dt)

        def callback(self, sender: str, elapsed: float, state_stack: Sequence[ProgressState]) -> None:
            pass

    def test_threaded(self):
        with self.assertRaises(ValueError) as e:
            self._TestThreaded(-1, 0)

        self.assertEqual("The timer's minimum must be >=0", str(e.exception))

        with self.assertRaises(ValueError) as e:
            self._TestThreaded(0, -1)

        self.assertEqual("The timer's time step must be >=0", str(e.exception))

        observer = self._TestThreaded()
        with self.assertRaises(ValueError) as e:
            observer.on_begin(state_stack=[])

        self.assertEqual('ProgressStates must be given', str(e.exception))

        with self.assertRaises(ValueError) as e:
            observer.on_update(state_stack=[])

        self.assertEqual('ProgressStates must be given', str(e.exception))

        with self.assertRaises(ValueError) as e:
            observer.on_end(state_stack=[])

        self.assertEqual('ProgressStates must be given', str(e.exception))


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
                   callback_config=dict(api_uri='https://xcube-gen.test/api/v1/jobs/tomtom/iamajob/callback',
                                 access_token='dfsvdfsv'))

    def setUp(self) -> None:
        self._request = Request.from_dict(self.REQUEST)
        self._callback_config = self._request.callback_config

    @requests_mock.Mocker()
    def test_observer(self, m):
        m.put('https://xcube-gen.test/api/v1/jobs/tomtom/iamajob/callback', json={})
        progress_state = ProgressState(label='test', total_work=0., super_work=10.)

        observer = CallbackApiProgressObserver(self._callback_config)
        observer.on_begin(state_stack=[progress_state])

        with self.assertRaises(ValueError) as e:
            self._callback_config.api_uri = None
            observer = CallbackApiProgressObserver(self._callback_config)
            observer.on_begin(state_stack=[progress_state])

        self.assertEqual('Both, api_uri and access_token must be given.', str(e.exception))

        with self.assertRaises(ValueError) as e:
            self._callback_config.access_token = None
            observer = CallbackApiProgressObserver(self._callback_config)
            observer.on_begin(state_stack=[progress_state])

        self.assertEqual('Both, api_uri and access_token must be given.', str(e.exception))


class TestCallbackTerminalProgressObserver(unittest.TestCase):
    def test_print_progress(self):
        CallbackTerminalProgressObserver(dt=0.1).activate()

        with observe_progress('Generating cube', 100) as cm:
            dt = 1
            for i in range(1, 80):
                cm.will_work(1)
                sleep(dt)
                cm.worked(1)

            cm.will_work(20)
            sleep(dt)
            cm.worked(20)
