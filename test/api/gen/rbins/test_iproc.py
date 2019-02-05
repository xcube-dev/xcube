import unittest

import numpy as np
import xarray as xr

from xcube.api.gen.rbins.iproc import RbinsSeviriHighrocDailyInputProcessor, \
    RbinsSeviriHighrocSceneInputProcessor


class RbinsSeviriHighrocSceneInputProcessorTest(unittest.TestCase):

    def setUp(self):
        self.processor = RbinsSeviriHighrocSceneInputProcessor()

    def test_props(self):
        self.assertEqual('rbins-seviri-highroc-scene-l2', self.processor.name)
        self.assertEqual('RBINS SEVIRI HIGHROC single-scene Level-2 NetCDF inputs', self.processor.description)
        self.assertEqual('netcdf4', self.processor.input_reader)

    def test_reprojection_info(self):
        reprojection_info = self.processor.get_reprojection_info(create_rbins_seviri_scene_dataset())
        self.assertEqual(('lon', 'lat'), reprojection_info.xy_var_names)
        self.assertEqual(1, reprojection_info.xy_gcp_step)

    def test_pre_process(self):
        ds1 = create_rbins_seviri_scene_dataset()
        ds2 = self.processor.pre_process(ds1)
        self.assertIs(ds1, ds2)

    def test_post_process(self):
        ds1 = create_rbins_seviri_scene_dataset()
        ds2 = self.processor.post_process(ds1)
        self.assertIs(ds1, ds2)


class RbinsSeviriHighrocDailyInputProcessorTest(unittest.TestCase):

    def setUp(self):
        self.processor = RbinsSeviriHighrocDailyInputProcessor()

    def test_props(self):
        self.assertEqual('rbins-seviri-highroc-daily-l2', self.processor.name)
        self.assertEqual('RBINS SEVIRI HIGHROC daily Level-2 NetCDF inputs', self.processor.description)
        self.assertEqual('netcdf4', self.processor.input_reader)

    def test_reprojection_info(self):
        reprojection_info = self.processor.get_reprojection_info(create_rbins_seviri_daily_dataset())
        self.assertEqual(('longitude', 'latitude'), reprojection_info.xy_var_names)
        self.assertEqual(1, reprojection_info.xy_gcp_step)

    def test_pre_process(self):
        ds1 = create_rbins_seviri_daily_dataset()
        ds2 = self.processor.pre_process(ds1)
        self.assertIsNot(ds1, ds2)
        self.assertIn('time', ds2)
        self.assertIn('latitude', ds2)
        self.assertIn('longitude', ds2)
        self.assertIn('SPM', ds2)
        self.assertIn('TUR', ds2)
        self.assertIn('DS_FLAGS', ds2)
        self.assertNotIn('DATE', ds2)
        self.assertNotIn('HOUR', ds2)
        self.assertNotIn('MIN', ds2)

    def test_post_process(self):
        ds1 = create_rbins_seviri_daily_dataset()
        ds2 = self.processor.post_process(ds1)
        self.assertIs(ds1, ds2)


def create_rbins_seviri_scene_dataset():
    """
    Simulates a HIGHROC OLCI L2 product in NetCDF 4 format
    """
    longitude = np.array([[8, 9.3, 10.6, 11.9],
                          [8, 9.2, 10.4, 11.6],
                          [8, 9.1, 10.2, 11.3]], dtype=np.float32)
    latitude = np.array([[56, 56.1, 56.2, 56.3],
                         [55, 55.2, 55.4, 55.6],
                         [54, 54.3, 54.6, 54.9]], dtype=np.float32)

    x = 4
    y = 3
    spm = np.zeros(shape=(y, x), dtype=np.float32)
    tur = np.zeros(shape=(y, x), dtype=np.float32)
    ds_flags = np.zeros(shape=(y, x), dtype=np.int32)
    return xr.Dataset(
        data_vars=dict(
            longitude=(('y', 'x'), longitude, dict(
                long_name="longitude",
                units="degrees_east",
            )),
            latitude=(('y', 'x'), latitude, dict(
                long_name="latitude",
                units="degrees_north",
            )),
            SPM=(('y', 'x'),
                 spm,
                 dict(units='mg/m3',
                      long_name='Suspended Particuate Matter',
                      algorithm='Nechad et al. 2010',
                      ds_flag='good value: FLAG = 0')),
            TUR=(('y', 'x'),
                 tur,
                 dict(units='FNU',
                      long_name='Turbidity',
                      algorithm='Neukermans et al. 2012',
                      ds_flag='good value: FLAG = 0')),
            DS_FLAGS=(('y', 'x'),
                      ds_flags),
        ),
        attrs=dict([('file_info', 'GRIMAS export NCDF file'),
                    ('processed_by', 'RBINS/DO Nature/REMSEM'),
                    ('generated', '06/08/2017 09:22:58'),
                    ('sensor', 'SEVIRI'),
                    ('REGION', 'SNS'),
                    ('SOURCE', 'EUR'),
                    ('U0', 29.3),
                    ('TU0M26', 0.98744),
                    ('TU0M28', 0.89171),
                    ('EPSDEF', 1.0),
                    ('EPSEST', 1),
                    ('RECAL', 'epsOffset'),
                    ('CALIBRATION', 0),
                    ('CLMMASK', 2),
                    ('RAY', 0),
                    ('LUTVERSION', 'DEFAULT'),
                    ('SWIRAC', 0),
                    ('NONLINEAR', 0),
                    ('NONLINEAR_TA', 0),
                    ('MEDIAN_RHOA', 0),
                    ('DNAME', 'QV_2013b.1'),
                    ('DATE', '20170806'),
                    ('TIME', '0700'),
                    ('SIGMA', 6.09),
                    ('EPS', 0.90903175),
                    ('EPSOFF', -0.0026553879),
                    ('PLATFORM', 'MSG3'),
                    ('DEPSOFF', 0.014735364),
                    ('DEPS', 0.0018116904),
                    ('NEPS', 1615),
                    ('ANCILLARY', 'defaults used')])
    )


def create_rbins_seviri_daily_dataset():
    """
    Simulates a HIGHROC OLCI L2 product in NetCDF 4 format
    """
    longitude = np.array([[8, 9.3, 10.6, 11.9],
                          [8, 9.2, 10.4, 11.6],
                          [8, 9.1, 10.2, 11.3]], dtype=np.float32)
    latitude = np.array([[56, 56.1, 56.2, 56.3],
                         [55, 55.2, 55.4, 55.6],
                         [54, 54.3, 54.6, 54.9]], dtype=np.float32)

    date = np.array([20170108, 20170108, 20170108, 20170108, 20170108, 20170108, 20170108,
                     20170108, 20170108, 20170108, 20170108, 20170108, 20170108, 20170108,
                     20170108, 20170108, 20170108, 20170108, 20170108, 20170108, 20170108,
                     20170108, 20170108, 20170108, 20170108, 20170108, 20170108, 20170108,
                     20170108, 20170108, 20170108, 20170108, 20170108, 20170108, 20170108,
                     20170108], dtype=np.int32)
    hour = np.array([7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11,
                     11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16], dtype=np.int32)
    minute = np.array([15, 30, 45, 0, 15, 30, 45, 0, 15, 30, 45, 0, 15, 30, 45, 0, 15, 30,
                       45, 0, 15, 30, 45, 0, 15, 30, 45, 0, 15, 30, 45, 0, 15, 30, 45, 0], dtype=np.int32)
    x = 4
    y = 3
    t = 36
    spm = np.zeros(shape=(t, y, x), dtype=np.float32)
    tur = np.zeros(shape=(t, y, x), dtype=np.float32)
    ds_flags = np.zeros(shape=(t, y, x), dtype=np.int32)
    return xr.Dataset(
        data_vars=dict(
            longitude=(('y', 'x'), longitude, dict(
                long_name="longitude",
                units="degrees_east",
            )),
            latitude=(('y', 'x'), latitude, dict(
                long_name="latitude",
                units="degrees_north",
            )),
            DATE=(('t',), date),
            HOUR=(('t',), hour),
            MIN=(('t',), minute),
            SPM=(('t', 'y', 'x'),
                 spm,
                 dict(units='mg/m3',
                      long_name='Suspended Particuate Matter',
                      algorithm='Nechad et al. 2010',
                      ds_flag='good value: FLAG = 0')),
            TUR=(('t', 'y', 'x'),
                 tur,
                 dict(units='FNU',
                      long_name='Turbidity',
                      algorithm='Neukermans et al. 2012',
                      ds_flag='good value: FLAG = 0')),
            DS_FLAGS=(('t', 'y', 'x'),
                      ds_flags),
        ),
        attrs=dict(processed_by='RBINS/DO Nature/REMSEM',
                   generated='2017-01-08 22:23',
                   sensor='SEVIRI',
                   region='SNS',
                   source='EUR',
                   platform='MSG3',
                   dname='QV_2013b.1')
    )
