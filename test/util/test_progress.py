import unittest

from xcube.util.progress import ProgressEmitter
from xcube.util.progress import ProgressMonitor


class ProgressMonitorTest(unittest.TestCase):

    def setUp(self) -> None:
        self.calls = []

    def _start(self, label, total_work):
        self.calls.append(('start', label, total_work))

    def _update(self, worked):
        self.calls.append(('update', worked))

    def _end(self, label, total_work):
        self.calls.append(('end'))

    def test_task_object(self):
        pm = ProgressMonitor(self._start, self._update, self._end)

        t = ProgressEmitter('test', 3)
        t.update(1)
        t.update(0.5)
        t.update(1.5)
        t.stop()

    def test_task_context_manager(self):
        with ProgressEmitter('test', 3) as t:
            t.update(1)
            t.update(0.5)
            t.update(1.5)
