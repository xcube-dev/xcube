import unittest
from xcube.util.progress import ProgressState


class TestProgress(unittest.TestCase):
    REQUEST = dict(input_configs=[dict(store_id='memory',
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
