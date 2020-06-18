import unittest
from typing import Sequence

from xcube.util.progress import ProgressObserver
from xcube.util.progress import ProgressState
from xcube.util.progress import observe_progress
from xcube.util.progress import observe_progress_local


class MyProgressObserver(ProgressObserver):
    def __init__(self):
        self.calls = []

    def on_begin(self, state_stack: Sequence[ProgressState]):
        self.calls.append(('begin', self._serialize_stack(state_stack)))

    def on_update(self, state_stack: Sequence[ProgressState]):
        self.calls.append(('update', self._serialize_stack(state_stack)))

    def on_end(self, state_stack: Sequence[ProgressState]):
        self.calls.append(('end', self._serialize_stack(state_stack)))

    @classmethod
    def _serialize_stack(cls, state_stack):
        return [(s.label, s.progress, s.finished) for s in state_stack]


class ProgressObserverTest(unittest.TestCase):

    def test_observe_progress(self):
        observer = MyProgressObserver()
        observer.activate()

        with observe_progress('computing', 4) as progress_reporter:
            # do something that takes 1 unit
            progress_reporter.worked(1)
            # do something that takes 1 unit
            progress_reporter.worked(1)
            # do something that takes 2 units
            progress_reporter.worked(2)

        self.assertEqual(
            [
                ('begin', [('computing', 0.0, False)]),
                ('update', [('computing', 0.25, False)]),
                ('update', [('computing', 0.5, False)]),
                ('update', [('computing', 1.0, False)]),
                ('end', [('computing', 1.0, True)])
            ],
            observer.calls)

    def test_nested_observe_progress(self):
        observer = MyProgressObserver()
        observer.activate()

        with observe_progress('computing', 4) as progress_reporter:
            # do something that takes 1 unit
            progress_reporter.worked(1)
            # do something that takes 1 unit
            progress_reporter.worked(1)
            # do something that will takes 2 units
            progress_reporter.will_work(2)
            with observe_progress('loading', 4) as progress_reporter_2:
                # do something that takes 3 units
                progress_reporter_2.worked(3)
                # do something that takes 1 unit
                progress_reporter_2.worked(1)

        self.assertEqual(
            [
                ('begin', [('computing', 0.0, False)]),
                ('update', [('computing', 0.25, False)]),
                ('update', [('computing', 0.5, False)]),
                ('begin', [('computing', 0.5, False), ('loading', 0.0, False)]),
                ('update', [('computing', 0.875, False), ('loading', 0.75, False)]),
                ('update', [('computing', 1.0, False), ('loading', 1.0, False)]),
                ('end', [('computing', 1.0, False), ('loading', 1.0, True)]),
                ('end', [('computing', 1.0, True)])
            ],
            observer.calls)
