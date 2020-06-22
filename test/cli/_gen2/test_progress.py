import unittest

from xcube.util.progress import ProgressState


class TestProgress(unittest.TestCase):
    def test_progress(self):
        progress = ProgressState(label='test', total_work=20, super_work=10)

        self.assertEqual('test', progress.label)
        self.assertEqual(20, progress.total_work)
        self.assertEqual(10, progress.super_work)
        self.assertEqual(0., progress.completed_work)
        self.assertEqual(1., progress.super_work_ahead)
        self.assertFalse(progress.finished)

        progress.inc_work(10)
        self.assertEqual(10, progress.completed_work)
        with self.assertRaises(ValueError) as e:
            progress.inc_work(-1)
        self.assertEqual('work must be greater than zero', str(e.exception))

        self.assertEqual(0.5, progress.progress)

        super_work = progress.to_super_work(10)
        self.assertEqual(5, super_work)

        progress.super_work_ahead = 10
        self.assertEqual(10, progress.super_work_ahead)
        with self.assertRaises(ValueError) as e:
            progress.super_work_ahead = -10
        self.assertEqual('work must be greater than zero', str(e.exception))

        progress.finish()
        self.assertTrue(progress.finished)



if __name__ == '__main__':
    unittest.main()
