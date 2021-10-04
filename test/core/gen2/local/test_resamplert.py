from xcube.core.new import new_cube
from xcube.core.gen2 import CubeConfig
from xcube.core.gen2.local.resamplert import CubeResamplerT
from xcube.core.gridmapping import GridMapping

import unittest


class CubeResamplerTTest(unittest.TestCase):

    @staticmethod
    def _get_cube(time_freq: str, time_periods: int):

        def b3(index1, index2, index3):
            return index1 + index2 * 0.1 + index3 * 0.01

        return new_cube(variables=dict(B03=b3),
                        time_periods=time_periods,
                        time_freq=time_freq,
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
        cube_config = CubeConfig(time_range=('2010-01-01', '2012-12-31'),
                                 time_period='1Y',
                                 temporal_resampling='min')
        temporal_resampler = CubeResamplerT(cube_config)

        cube = self._get_cube(time_freq='M', time_periods=12)

        resampled_cube, grid_mapping, cube_config = temporal_resampler.\
            transform_cube(cube,
                           GridMapping.from_dataset(cube),
                           cube_config)
        self.assertEquals(cube, resampled_cube)

    def test_transform_cube_downsample_to_months(self):
        cube_config = CubeConfig(time_range=('2010-08-01', '2010-11-30'),
                                 time_period='1M',
                                 temporal_resampling='min')
        temporal_resampler = CubeResamplerT(cube_config)

        cube = self._get_cube(time_freq='W', time_periods=12)

        resampled_cube, grid_mapping, cube_config = temporal_resampler.\
            transform_cube(cube,
                           GridMapping.from_dataset(cube),
                           cube_config)
        self.assertEquals(cube, resampled_cube)

    def test_transform_cube_downsample_to_weeks(self):
        cube_config = CubeConfig(time_range=('2010-08-03', '2010-09-10'),
                                 time_period='2W',
                                 temporal_resampling='min')
        temporal_resampler = CubeResamplerT(cube_config)

        cube = self._get_cube(time_freq='D', time_periods=22)

        resampled_cube, grid_mapping, cube_config = temporal_resampler.\
            transform_cube(cube,
                           GridMapping.from_dataset(cube),
                           cube_config)
        self.assertEquals(cube, resampled_cube)

