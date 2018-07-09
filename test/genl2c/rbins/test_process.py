import os
import unittest

from test.genl2c.rbins.helpers import get_inputdata_path
from xcube.genl2c.process import generate_l2c_cube
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
            input_files=[get_inputdata_path('SEVIRI_SNS_EUR_201708060700_QV_2013b.1.nc.gz')],
            output_name='l2c-single',
            output_format='netcdf4',
            append_mode=False)
        self.assertEqual(True, status)
        self.assertEqual(os.path.join('.', 'l2c-single.nc'), path)

    def _test_process_inputs_append_multiple_nc(self):
        # FIXME: this test still fails
        path, status = process_inputs_wrapper(
            input_files=[get_inputdata_path('SEVIRI_SNS_EUR_20170806????_QV_2013b.1.nc.gz')],
            output_name='l2c',
            output_format='netcdf4',
            append_mode=True)
        self.assertEqual(True, status)
        self.assertEqual(os.path.join('.', 'l2c.nc'), path)

    def test_process_inputs_append_multiple_zarr(self):
        path, status = process_inputs_wrapper(
            input_files=[get_inputdata_path('SEVIRI_SNS_EUR_20170806????_QV_2013b.1.nc.gz')],
            output_name='l2c',
            output_format='zarr',
            append_mode=True)
        self.assertEqual(True, status)
        self.assertEqual(os.path.join('.', 'l2c.zarr'), path)


# noinspection PyShadowingBuiltins
def process_inputs_wrapper(input_files=None,
                           output_name=None,
                           output_format='netcdf4',
                           append_mode=False):
    return generate_l2c_cube(input_files=input_files,
                             input_type='rbins-seviri-highroc-scene-l2',
                             output_size=(320, 180),
                             output_region=(-4., 47., 12., 56.),
                             output_resampling='Nearest',
                             output_variables=['KPAR', 'SPM', 'TUR'],
                             output_dir='.',
                             output_name=output_name,
                             output_format=output_format,
                             append_mode=append_mode,
                             dry_run=False,
                             monitor=None)
