import unittest
from typing import Sequence

from xcube.util.progress import ProgressObserver
from xcube.util.progress import ProgressState
from xcube.util.progress import add_progress_observers
from xcube.util.progress import new_progress_observers
from xcube.util.progress import observe_progress


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


class ObserveProgressTest(unittest.TestCase):

    def test_observe_progress(self):
        observer = MyProgressObserver()
        observer.activate()

        with observe_progress('computing', 4) as reporter:
            # do something that takes 1 unit
            reporter.worked(1)
            # do something that takes 1 unit
            reporter.worked(1)
            # do something that takes 2 units
            reporter.worked(2)

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

        with observe_progress('computing', 4) as reporter:
            # do something that takes 1 unit
            reporter.worked(1)
            # do something that takes 1 unit
            reporter.worked(1)
            # do something that will take 2 units
            reporter.will_work(2)
            with observe_progress('loading', 4) as nested_reporter:
                # do something that takes 3 units
                nested_reporter.worked(3)
                # do something that takes 1 unit
                nested_reporter.worked(1)

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

    def test_nested_observe_progress_with_new_progress_observers(self):
        observer = MyProgressObserver()
        observer.activate()

        nested_observer = MyProgressObserver()

        with observe_progress('computing', 4) as progress_reporter:
            # do something that takes 1 unit
            progress_reporter.worked(1)
            # do something that takes 1 unit
            progress_reporter.worked(1)
            with new_progress_observers(nested_observer):
                with observe_progress('loading', 4) as progress_reporter_2:
                    # do something that takes 3 units
                    progress_reporter_2.worked(3)
                    # do something that takes 1 unit
                    progress_reporter_2.worked(1)

            # do something that takes 1 unit
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

        self.assertEqual(
            [
                ('begin', [('loading', 0.0, False)]),
                ('update', [('loading', 0.75, False)]),
                ('update', [('loading', 1.0, False)]),
                ('end', [('loading', 1.0, True)])
            ],
            nested_observer.calls)

    def test_nested_observe_progress_with_add_progress_observers(self):
        observer = MyProgressObserver()
        observer.activate()

        nested_observer = MyProgressObserver()

        with observe_progress('computing', 4) as reporter:
            # do something that takes 1 unit
            reporter.worked(1)
            # do something that takes 1 unit
            reporter.worked(1)
            # do something that will take 2 units
            reporter.will_work(2)
            with add_progress_observers(nested_observer):
                with observe_progress('loading', 4) as nested_reported:
                    # do something that takes 3 units
                    nested_reported.worked(3)
                    # do something that takes 1 unit
                    nested_reported.worked(1)

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

        self.assertEqual(
            [
                ('begin', [('computing', 0.5, False), ('loading', 0.0, False)]),
                ('update', [('computing', 0.875, False), ('loading', 0.75, False)]),
                ('update', [('computing', 1.0, False), ('loading', 1.0, False)]),
                ('end', [('computing', 1.0, False), ('loading', 1.0, True)])
            ],
            nested_observer.calls)


class ProgressStateTest(unittest.TestCase):

    def test_progress_state_props(self):
        state = ProgressState('computing', 100, 3)
        self.assertEqual('computing', state.label)
        self.assertEqual(100, state.total_work)
        self.assertEqual(3, state.super_work)
        self.assertEqual(1, state.super_work_ahead)
        self.assertEqual(0, state.completed_work)
        self.assertEqual(0, state.progress)
        self.assertEqual(0.6, state.to_super_work(20))
        state.inc_work(15)
        self.assertEqual(15, state.completed_work)
        self.assertEqual(0.15, state.progress)
        self.assertEqual(0.6, state.to_super_work(20))

