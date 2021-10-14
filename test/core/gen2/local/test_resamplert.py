from xcube.core.new import new_cube
from xcube.core.gen2 import CubeConfig
from xcube.core.gen2.local.resamplert import CubeResamplerT
from xcube.core.gridmapping import GridMapping

import cftime
import numpy as np
import unittest


class CubeResamplerTTest(unittest.TestCase):

    @staticmethod
    def _get_cube(time_freq: str, time_periods: int, use_cftime: bool = False):

        def b3(index1, index2, index3):
            return index1 + index2 * 0.1 + index3 * 0.01

        return new_cube(variables=dict(B03=b3),
                        time_periods=time_periods,
                        time_freq=time_freq,
                        use_cftime=use_cftime,
                        time_dtype= 'datetime64[s]' if not use_cftime else None,
                        width=10, height=5, time_start='2010-08-04')

    def test_transform_cube_no_time_period(self):
        cube_config = CubeConfig(time_range=('2010-01-01', '2012-12-31'))
        temporal_resampler = CubeResamplerT(cube_config)

        cube = self._get_cube(time_freq='M', time_periods=12)

        resampled_cube, grid_mapping, cube_config = temporal_resampler.\
            transform_cube(cube,
                           GridMapping.from_dataset(cube),
                           cube_config)
        self.assertEquals(cube, resampled_cube)

    def test_transform_cube_downsample_to_years(self):
        cube_config = CubeConfig(time_range=('2010-01-01', '2014-12-31'),
                                 time_period='2Y',
                                 temporal_resampling=dict(downsampling='min'))
        temporal_resampler = CubeResamplerT(cube_config)

        cube = self._get_cube(time_freq='M', time_periods=24)

        resampled_cube, grid_mapping, cube_config = temporal_resampler.\
            transform_cube(cube,
                           GridMapping.from_dataset(cube),
                           cube_config)
        self.assertIsNotNone(resampled_cube)
        np.testing.assert_equal(
            resampled_cube.time.values,
            np.array(['2011-01-01T00:00:00', '2013-01-01T00:00:00'],
                     dtype=np.datetime64))
        np.testing.assert_equal(
            resampled_cube.time_bnds.values,
            np.array([['2010-01-01T00:00:00', '2012-01-01T00:00:00'],
                      ['2012-01-01T00:00:00', '2014-01-01T00:00:00']],
                     dtype=np.datetime64))
        self.assertEquals((2, 5, 10), resampled_cube.B03.shape)
        self.assertAlmostEquals(0.0, resampled_cube.B03[0].values.min(), 8)
        self.assertAlmostEquals(16.0, resampled_cube.B03[1].values.min(), 8)

    def test_transform_cube_downsample_to_months(self):
        cube_config = CubeConfig(time_range=('2010-08-01', '2010-11-30'),
                                 time_period='2M',
                                 temporal_resampling='min')
        temporal_resampler = CubeResamplerT(cube_config)

        cube = self._get_cube(time_freq='W', time_periods=12)

        resampled_cube, grid_mapping, cube_config = temporal_resampler.\
            transform_cube(cube,
                           GridMapping.from_dataset(cube),
                           cube_config)
        self.assertIsNotNone(resampled_cube)
        np.testing.assert_equal(
            resampled_cube.time.values,
            np.array(['2010-09-01T00:00:00', '2010-11-01T00:00:00'],
                     dtype=np.datetime64))
        np.testing.assert_equal(
            resampled_cube.time_bnds.values,
            np.array([['2010-08-01T00:00:00', '2010-10-01T00:00:00'],
                      ['2010-10-01T00:00:00', '2010-12-01T00:00:00']],
                     dtype=np.datetime64))
        self.assertEquals((2, 5, 10), resampled_cube.B03.shape)
        self.assertAlmostEquals(0.0, resampled_cube.B03[0].values.min(), 8)
        self.assertAlmostEquals(8.0, resampled_cube.B03[1].values.min(), 8)

    def test_transform_cube_downsample_to_weeks(self):
        cube_config = CubeConfig(time_range=('2010-08-03', '2010-09-10'),
                                 time_period='2W',
                                 temporal_resampling='max')
        temporal_resampler = CubeResamplerT(cube_config)

        cube = self._get_cube(time_freq='D', time_periods=32)

        resampled_cube, grid_mapping, cube_config = temporal_resampler.\
            transform_cube(cube,
                           GridMapping.from_dataset(cube),
                           cube_config)
        self.assertIsNotNone(resampled_cube)
        np.testing.assert_equal(
            resampled_cube.time.values,
            np.array(['2010-08-08T00:00:00', '2010-08-22T00:00:00',
                      '2010-09-05T00:00:00'],
                     dtype=np.datetime64))
        np.testing.assert_equal(
            resampled_cube.time_bnds.values,
            np.array([['2010-08-01T00:00:00', '2010-08-15T00:00:00'],
                      ['2010-08-15T00:00:00', '2010-08-29T00:00:00'],
                      ['2010-08-29T00:00:00', '2010-09-12T00:00:00']],
                     dtype=np.datetime64))
        self.assertEquals((3, 5, 10), resampled_cube.B03.shape)
        self.assertAlmostEquals(10.0, resampled_cube.B03[0].values.min(), 8)
        self.assertAlmostEquals(24.0, resampled_cube.B03[1].values.min(), 8)
        self.assertAlmostEquals(31.0, resampled_cube.B03[2].values.min(), 8)

    def test_transform_cube_upsample_to_months(self):
        cube_config = CubeConfig(time_range=('2011-10-01', '2012-03-31'),
                                 time_period='2M',
                                 temporal_resampling='linear')
        temporal_resampler = CubeResamplerT(cube_config)

        cube = self._get_cube(time_freq='Y', time_periods=2)

        resampled_cube, grid_mapping, cube_config = temporal_resampler.\
            transform_cube(cube,
                           GridMapping.from_dataset(cube),
                           cube_config)
        self.assertIsNotNone(resampled_cube)
        np.testing.assert_equal(
            resampled_cube.time.values,
            np.array(['2011-11-01T00:00:00', '2012-01-01T00:00:00',
                      '2012-03-01T00:00:00'],
                     dtype=np.datetime64))
        np.testing.assert_equal(
            resampled_cube.time_bnds.values,
            np.array([['2011-10-01T00:00:00', '2011-12-01T00:00:00'],
                      ['2011-12-01T00:00:00', '2012-02-01T00:00:00'],
                      ['2012-02-01T00:00:00', '2012-04-01T00:00:00']],
                     dtype=np.datetime64))
        self.assertEquals((3, 5, 10), resampled_cube.B03.shape)
        self.assertAlmostEquals(0.33561644,
                                resampled_cube.B03[0].values.min(), 8)
        self.assertAlmostEquals(0.50273973,
                                resampled_cube.B03[1].values.min(), 8)
        self.assertAlmostEquals(0.66712329,
                                resampled_cube.B03[2].values.min(), 8)

    def test_transform_cube_upsample_to_weeks(self):
        cube_config = CubeConfig(time_range=('2010-09-01', '2010-10-10'),
                                 time_period='4W',
                                 temporal_resampling='nearest')
        temporal_resampler = CubeResamplerT(cube_config)

        cube = self._get_cube(time_freq='M', time_periods=4)

        resampled_cube, grid_mapping, cube_config = temporal_resampler.\
            transform_cube(cube,
                           GridMapping.from_dataset(cube),
                           cube_config)
        self.assertIsNotNone(resampled_cube)
        np.testing.assert_equal(
            resampled_cube.time.values,
            np.array(['2010-09-12T00:00:00', '2010-10-10T00:00:00'],
                     dtype=np.datetime64))
        np.testing.assert_equal(
            resampled_cube.time_bnds.values,
            np.array([['2010-08-29T00:00:00', '2010-09-26T00:00:00'],
                      ['2010-09-26T00:00:00', '2010-10-24T00:00:00']],
                     dtype=np.datetime64))
        self.assertEquals((2, 5, 10), resampled_cube.B03.shape)
        self.assertAlmostEquals(0.0, resampled_cube.B03[0].values.min(), 8)
        self.assertAlmostEquals(1.0, resampled_cube.B03[1].values.min(), 8)

    def test_transform_cube_upsample_to_days(self):
        cube_config = CubeConfig(time_range=('2010-08-14', '2010-08-24'),
                                 time_period='2D',
                                 temporal_resampling='linear')
        temporal_resampler = CubeResamplerT(cube_config)

        cube = self._get_cube(time_freq='W', time_periods=3)

        resampled_cube, grid_mapping, cube_config = temporal_resampler.\
            transform_cube(cube,
                           GridMapping.from_dataset(cube),
                           cube_config)
        self.assertIsNotNone(resampled_cube)
        np.testing.assert_equal(
            resampled_cube.time.values,
            np.array(['2010-08-15T00:00:00', '2010-08-17T00:00:00',
                      '2010-08-19T00:00:00', '2010-08-21T00:00:00',
                      '2010-08-23T00:00:00'],
                     dtype=np.datetime64))
        np.testing.assert_equal(
            resampled_cube.time_bnds.values,
            np.array([['2010-08-14T00:00:00', '2010-08-16T00:00:00'],
                      ['2010-08-16T00:00:00', '2010-08-18T00:00:00'],
                      ['2010-08-18T00:00:00', '2010-08-20T00:00:00'],
                      ['2010-08-20T00:00:00', '2010-08-22T00:00:00'],
                      ['2010-08-22T00:00:00', '2010-08-24T00:00:00']],
                     dtype=np.datetime64))
        self.assertEquals((5, 5, 10), resampled_cube.B03.shape)
        self.assertAlmostEquals(0.5,
                                resampled_cube.B03[0].values.min(), 8)
        self.assertAlmostEquals(0.78571429,
                                resampled_cube.B03[1].values.min(), 8)
        self.assertAlmostEquals(1.07142857,
                                resampled_cube.B03[2].values.min(), 8)
        self.assertAlmostEquals(1.35714286,
                                resampled_cube.B03[3].values.min(), 8)
        self.assertAlmostEquals(1.64285714,
                                resampled_cube.B03[4].values.min(), 8)

    def test_transform_cube_downsample_to_years_cftimes(self):
        cube_config = CubeConfig(time_range=('2010-01-01', '2014-12-31'),
                                 time_period='2Y',
                                 temporal_resampling='min')
        temporal_resampler = CubeResamplerT(cube_config)

        cube = self._get_cube(time_freq='M', time_periods=24, use_cftime=True)

        resampled_cube, grid_mapping, cube_config = temporal_resampler.\
            transform_cube(cube,
                           GridMapping.from_dataset(cube),
                           cube_config)
        self.assertIsNotNone(resampled_cube)
        np.testing.assert_equal(resampled_cube.time.values,
                                [cftime.DatetimeProlepticGregorian(2011, 1, 1),
                                 cftime.DatetimeProlepticGregorian(2013, 1, 1)])
        np.testing.assert_equal(
            resampled_cube.time_bnds.values,
            [[cftime.DatetimeProlepticGregorian(2010, 1, 1),
             cftime.DatetimeProlepticGregorian(2012, 1, 1)],
             [cftime.DatetimeProlepticGregorian(2012, 1, 1),
              cftime.DatetimeProlepticGregorian(2014, 1, 1)]])
        self.assertEquals((2, 5, 10), resampled_cube.B03.shape)
        self.assertAlmostEquals(0.0, resampled_cube.B03[0].values.min(), 8)
        self.assertAlmostEquals(16.0, resampled_cube.B03[1].values.min(), 8)

    def test_transform_cube_upsample_to_months_cftimes(self):
        cube_config = CubeConfig(time_range=('2011-10-01', '2012-03-31'),
                                 time_period='2M',
                                 temporal_resampling='linear')
        temporal_resampler = CubeResamplerT(cube_config)

        cube = self._get_cube(time_freq='Y', time_periods=2, use_cftime=True)

        resampled_cube, grid_mapping, cube_config = temporal_resampler.\
            transform_cube(cube,
                           GridMapping.from_dataset(cube),
                           cube_config)
        self.assertIsNotNone(resampled_cube)
        np.testing.assert_equal(
            resampled_cube.time.values,
            [cftime.DatetimeProlepticGregorian(2011, 11, 1),
             cftime.DatetimeProlepticGregorian(2012, 1, 1),
             cftime.DatetimeProlepticGregorian(2012, 3, 1)])
        np.testing.assert_equal(
            resampled_cube.time_bnds.values,
            [[cftime.DatetimeProlepticGregorian(2011, 10, 1),
              cftime.DatetimeProlepticGregorian(2011, 12, 1)],
             [cftime.DatetimeProlepticGregorian(2011, 12, 1),
              cftime.DatetimeProlepticGregorian(2012, 2, 1)],
             [cftime.DatetimeProlepticGregorian(2012, 2, 1),
              cftime.DatetimeProlepticGregorian(2012, 4, 1)]])
        self.assertEquals((3, 5, 10), resampled_cube.B03.shape)
        self.assertAlmostEquals(0.33561644,
                                resampled_cube.B03[0].values.min(), 8)
        self.assertAlmostEquals(0.50273973,
                                resampled_cube.B03[1].values.min(), 8)
        self.assertAlmostEquals(0.66712329,
                                resampled_cube.B03[2].values.min(), 8)
