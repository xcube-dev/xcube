import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal
import xarray as xr

from xcube.genl2c.rbins.inputprocessor import RbinsSeviriInputProcessor, init_plugin

nan = np.nan


class RbinsSeviriInputProcessorTest(unittest.TestCase):

    def setUp(self):
        self.processor = RbinsSeviriInputProcessor()

    def test_props(self):
        self.assertEqual('rbins-seviri-l2', self.processor.name)
        self.assertEqual('RBINS SEVIRI Level-2 NetCDF inputs', self.processor.description)
        self.assertEqual({'r'}, self.processor.modes)
        self.assertEqual('nc', self.processor.ext)
        self.assertIsNotNone(self.processor.input_info)
        self.assertEqual(('longitude', 'latitude'), self.processor.input_info.xy_var_names)
        self.assertEqual(1, self.processor.input_info.xy_gcp_step)
        self.assertEqual(None, self.processor.input_info.time_var_name)

    def test_read(self):
        with self.assertRaises(OSError):
            self.processor.read('test-nc')

    def test_pre_reproject(self):
        ds1 = create_rbins_seviri_dataset()
        ds2 = self.processor.pre_reproject(ds1)
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

    def test_post_reproject(self):
        ds1 = create_rbins_seviri_dataset()
        ds2 = self.processor.post_reproject(ds1)
        self.assertIs(ds1, ds2)

    def test_init_plugin(self):
        init_plugin()

def create_rbins_seviri_dataset():
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
                units="degrees east",
            )),
            latitude=(('y', 'x'), latitude, dict(
                long_name="latitude",
                units="degrees north",
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
