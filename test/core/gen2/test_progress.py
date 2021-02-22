import unittest
from unittest.mock import patch

import requests_mock

from xcube.core.gen2.request import CubeGeneratorRequest
from xcube.core.gen2.progress import ApiProgressCallbackObserver
from xcube.core.gen2.progress import TerminalProgressCallbackObserver
from xcube.core.gen2.progress import _ThreadedProgressObserver
from xcube.util.progress import ProgressState


class TestThreadedProgressObservers(unittest.TestCase):
    REQUEST = dict(input_config=dict(store_id='sentinelhub',
                                     data_id='S2L2A',
                                     ),
                   cube_config=dict(variable_names=['B01', 'B02', 'B03'],
                                    crs='WGS84',
                                    bbox=[12.2, 52.1, 13.9, 54.8],
                                    spatial_res=0.05,
                                    time_range=['2018-01-01', None],
                                    time_period='4D'),
                   output_config=dict(store_id='memory',
                                      data_id='CHL'),
                   callback_config=dict(api_uri='https://xcube-gen.test/api/v1/jobs/tomtom/iamajob/callback',
                                        access_token='dfsvdfsv'))

    def setUp(self) -> None:
        self._request = CubeGeneratorRequest.from_dict(self.REQUEST)
        self._callback_config = self._request.callback_config

    @requests_mock.Mocker()
    def test_api_delegate(self, m):
        m.put('https://xcube-gen.test/api/v1/jobs/tomtom/iamajob/callback', json={})
        progress_state = ProgressState(label='test', total_work=0., super_work=10.)

        observer = ApiProgressCallbackObserver(self._callback_config)
        observer.on_begin(state_stack=[progress_state])

        with self.assertRaises(ValueError) as e:
            self._callback_config.api_uri = None
            observer = ApiProgressCallbackObserver(self._callback_config)
            observer.on_begin(state_stack=[progress_state])

        self.assertEqual('Both, api_uri and access_token must be given.', str(e.exception))

        with self.assertRaises(ValueError) as e:
            self._callback_config.access_token = None
            observer = ApiProgressCallbackObserver(self._callback_config)
            observer.on_begin(state_stack=[progress_state])

        self.assertEqual('Both, api_uri and access_token must be given.', str(e.exception))

    @requests_mock.Mocker()
    def test_callback(self, m):
        expected_callback = {
            "sender": "on_begin",
            "state": {
                "label": "Test",
                "total_work": 100,
                "error": False,
                "progress": 0.0,
                "elapsed": 3.,
            }
        }

        m.put('https://xcube-gen.test/api/v1/jobs/tomtom/iamajob/callback', json=expected_callback)

        state_stack = ProgressState('Test', 100, 100)

        observer = ApiProgressCallbackObserver(self._callback_config)

        res = observer.callback("on_begin", 3., [state_stack])
        self.assertDictEqual(expected_callback, res.request.json())

        with self.assertRaises(ValueError) as e:
            observer.callback("on_begin", 3., [])

        self.assertEqual("ProgressStates must be given", str(e.exception))

        observer = TerminalProgressCallbackObserver()
        res = observer.callback("on_begin", 3., [state_stack], False)
        self.assertIn('Test', res)
        self.assertIn('0% Completed', res)

    def test_threaded_progress_on_begin(self):
        _mock_patch = patch('xcube.core.gen2.progress._ThreadedProgressObserver._start_timer')
        _mock = _mock_patch.start()

        observer = TerminalProgressCallbackObserver()

        state_stack = ProgressState('Test', 100, 100)

        observer.on_begin([state_stack])
        self.assertTrue(_mock.called)
        _mock.stop()

        _mock = _mock_patch.start()
        state_stack1 = ProgressState('Test', 100, 100)
        state_stack2 = ProgressState('Test', 100, 100)

        observer.on_begin([state_stack1, state_stack2])
        self.assertFalse(_mock.called)

    def test_threaded_progress_on_end(self):
        _mock_patch = patch('xcube.core.gen2.progress._ThreadedProgressObserver._stop_timer')
        _mock = _mock_patch.start()

        observer = TerminalProgressCallbackObserver()
        observer._running = True

        state_stack = ProgressState('Test', 100, 100)

        observer.on_end([state_stack])
        self.assertTrue(_mock.called)
        _mock.stop()

        _mock = _mock_patch.start()
        state_stack1 = ProgressState('Test', 100, 100)
        state_stack2 = ProgressState('Test', 100, 100)

        observer.on_end([state_stack1, state_stack2])
        self.assertFalse(_mock.called)

    # def test_running_progress(self):
    #     """
    #     Uncomment the lines below if you want to run and test the termial progress bar output.
    #     """

    # from time import sleep
    # from xcube.util.progress import observe_progress
    #
    # TerminalProgressCallbackObserver().activate()
    #
    # with observe_progress('Generating cube', 100) as cm:
    #     dt = 1
    #     for i in range(1, 80):
    #         cm.will_work(1)
    #         sleep(dt)
    #         cm.worked(1)
    #
    #     cm.will_work(20)
    #     sleep(dt)
    #     cm.worked(20)


class TestThreadedProgressObserver(unittest.TestCase):
    def test_threaded(self):
        with self.assertRaises(ValueError) as e:
            _ThreadedProgressObserver(minimum=-1, dt=0)

        self.assertEqual("The timer's minimum must be >=0", str(e.exception))

        with self.assertRaises(ValueError) as e:
            _ThreadedProgressObserver(minimum=0, dt=-1)

        self.assertEqual("The timer's time step must be >=0", str(e.exception))

        observer = _ThreadedProgressObserver(minimum=0, dt=0)

        with self.assertRaises(ValueError) as e:
            observer.on_begin(state_stack=[])

        self.assertEqual('state_stack must be given', str(e.exception))

        with self.assertRaises(ValueError) as e:
            observer.on_update(state_stack=[])

        self.assertEqual('state_stack must be given', str(e.exception))

        with self.assertRaises(ValueError) as e:
            observer.on_end(state_stack=[])

        self.assertEqual('state_stack must be given', str(e.exception))
