import os
import unittest

from test.genl2c.rbins.helpers import get_inputdata_path
from xcube.genl2c.process import process_inputs
from xcube.io import rimraf


def clean_up():
    files = ['l2c-single.nc', 'l2c.nc', 'l2c.zarr']
    for file in files:
        rimraf(os.path.join('.', file))
        rimraf(os.path.join('.', file + 'temp.nc'))


class RbinsProcessTest(unittest.TestCase):

    def setUp(self):
        clean_up()

    def tearDown(self):
        clean_up()

    # noinspection PyMethodMayBeStatic
    def test_process_inputs_single(self):
        path, status = process_inputs_wrapper(
            input=[get_inputdata_path('SEVIRI_SNS_EUR_201708060700_QV_2013b.1.nc.gz')],
            name='l2c-single',
            format='netcdf4',
            append=False)
        self.assertEqual(True, status)
        self.assertEqual(os.path.join('.', 'l2c-single.nc'), path)

    def _test_process_inputs_append_multiple_nc(self):
        # FIXME: this test still fails
        path, status = process_inputs_wrapper(
            input=[get_inputdata_path('SEVIRI_SNS_EUR_20170806????_QV_2013b.1.nc.gz')],
            name='l2c',
            format='netcdf4',
            append=True)
        self.assertEqual(True, status)
        self.assertEqual(os.path.join('.', 'l2c.nc'), path)

    def test_process_inputs_append_multiple_zarr(self):
        path, status = process_inputs_wrapper(
            input=[get_inputdata_path('SEVIRI_SNS_EUR_20170806????_QV_2013b.1.nc.gz')],
            name='l2c',
            format='zarr',
            append=True)
        self.assertEqual(True, status)
        self.assertEqual(os.path.join('.', 'l2c.zarr'), path)


# noinspection PyShadowingBuiltins
def process_inputs_wrapper(input=None, name=None, format='netcdf4', append=False):
    return process_inputs(input,
                          'rbins-seviri-highroc-scene-l2',
                          (320, 180),
                          (-4., 47., 12., 56.),
                          None,
                          None,
                          '.',
                          name,
                          output_format=format,
                          append=append, dry_run=False, monitor=None)
