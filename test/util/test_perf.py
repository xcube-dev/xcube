import time
from unittest import TestCase

from xcube.util.perf import measure_time_cm


class MockLogger:
    def __init__(self):
        self.output = []

    # noinspection PyUnusedLocal
    def info(self, msg: str, *args, **kwargs):
        if len(args) == 1:
            args = args[0]
        self.output.append(msg % args)


class MeasureTimeTest(TestCase):
    def test_enabled(self):
        logger = MockLogger()
        measure_time = measure_time_cm(disabled=False, logger=logger)

        with measure_time("Hello") as cm:
            time.sleep(0.06)
        self.assertTrue(hasattr(cm, "duration"))
        self.assertTrue(cm.duration > 0.05)
        self.assertIs(logger, cm.logger)
        self.assertEqual("Hello", cm.message)
        self.assertEqual(1, len(logger.output))
        self.assertTrue(logger.output[0].startswith("Hello: took "))

        with measure_time("Hello %s", "Mrs X") as cm:
            time.sleep(0.06)
        self.assertTrue(hasattr(cm, "duration"))
        self.assertTrue(cm.duration > 0.05)
        self.assertIsNotNone(cm.logger)
        self.assertEqual("Hello %s", cm.message)
        self.assertEqual(("Mrs X",), cm.args)
        self.assertEqual(2, len(logger.output))
        self.assertTrue(logger.output[1].startswith("Hello Mrs X: took "))

    def test_enabled_deprecated(self):
        measure_time = measure_time_cm(disabled=False)
        with measure_time(tag="Hello") as cm:
            time.sleep(0.06)
        self.assertTrue(hasattr(cm, "duration"))
        self.assertTrue(cm.duration > 0.05)
        self.assertIsNotNone(cm.logger)
        self.assertEqual("Hello", cm.message)

    def test_disabled(self):
        measure_time = measure_time_cm(disabled=True)
        with measure_time("hello") as cm:
            time.sleep(0.05)
        self.assertTrue(hasattr(cm, "duration"))
        self.assertIsNone(cm.duration)
        self.assertIsNone(cm.logger)
