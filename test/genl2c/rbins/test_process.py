import os
import unittest

from test.genl2c.rbins.helpers import get_inputdata_path
from xcube.genl2c.process import process_inputs
from xcube.io import rimraf


class RbinsProcessTest(unittest.TestCase):
    # noinspection PyMethodMayBeStatic
    def test_process_inputs_single(self):
        path, status = process_inputs_wrapper(input=[get_inputdata_path('SEVIRI_SNS_EUR_201708060700_QV_2013b.1.nc.gz')],
                                              name='l2c-single',
                                              format='netcdf4',
                                              append=False)
        self.assertEqual(True, status)
        self.assertEqual(os.path.join('.', 'l2c-single.nc'), path)
        rimraf(os.path.join('.', 'l2c-single.nc'))

    def test_process_inputs_append_multiple_nc(self):
        path, status = process_inputs_wrapper(input=[get_inputdata_path('SEVIRI_SNS_EUR_20170806????_QV_2013b.1.nc.gz')],
                                              name='l2c',
                                              format='netcdf4',
                                              append=True)
        self.assertEqual(True, status)
        self.assertEqual(os.path.join('.', 'l2c.nc'), path)
        rimraf(os.path.join('.', 'l2c.nc'))

    def test_process_inputs_append_multiple_zarr(self):
        path, status = process_inputs_wrapper(input=[get_inputdata_path('SEVIRI_SNS_EUR_20170806????_QV_2013b.1.nc.gz')],
                                              name='l2c',
                                              format='zarr',
                                              append=True)
        self.assertEqual(True, status)
        self.assertEqual(os.path.join('.', 'l2c.zarr'), path)
        rimraf(os.path.join('.', 'l2c.zarr'))



# noinspection PyShadowingBuiltins
def process_inputs_wrapper(input=None, name=None, format='netcdf4', append=False):
    return process_inputs(input,
                          'rbins-seviri-highroc-l2',
                          (320, 180),
                          (-4., 47., 12., 58.),
                          None,
                          None,
                          '.',
                          name,
                          output_format=format,
                          append=append, dry_run=False, monitor=None)
