import os
import unittest

from test.genl2c.snap.helpers import get_inputdata_file
from xcube.dsio import rimraf
from xcube.genl2c.process import generate_l2c_cube


def clean_up():
    files = ['l2c-single.nc', 'l2c.nc', 'l2c.zarr']
    for file in files:
        rimraf(os.path.join('.', file))
        rimraf(os.path.join('.', file + 'temp.nc'))


class SnapProcessTest(unittest.TestCase):

    def setUp(self):
        clean_up()

    def tearDown(self):
        clean_up()

    # noinspection PyMethodMayBeStatic
    def test_process_inputs_single(self):
        path, status = process_inputs_wrapper(input_files=[get_inputdata_file('O_L2_0001_SNS_2017105100139_v1.0.nc')],
                                              output_name='l2c-single',
                                              output_format='netcdf4',
                                              append_mode=False)
        self.assertEqual(True, status)
        self.assertEqual(os.path.join('.', 'l2c-single.nc'), path)

    def test_process_inputs_append_multiple_nc(self):
        path, status = process_inputs_wrapper(input_files=[get_inputdata_file('O_L2_0001_SNS_*_v1.0.nc')],
                                              output_name='l2c',
                                              output_format='netcdf4',
                                              append_mode=True)
        self.assertEqual(True, status)
        self.assertEqual(os.path.join('.', 'l2c.nc'), path)

    def test_process_inputs_append_multiple_zarr(self):
        path, status = process_inputs_wrapper(input_files=[get_inputdata_file('O_L2_0001_SNS_*_v1.0.nc')],
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
                             input_type='snap-olci-highroc-l2',
                             output_size=(2000, 1000),
                             output_region=(0., 50., 5., 52.5),
                             output_variables=[('conc_chl', None), ('conc_tsm', None), ('kd489', None)],
                             output_resampling='Nearest',
                             output_dir='.',
                             output_name=output_name,
                             output_format=output_format,
                             append_mode=append_mode,
                             dry_run=False,
                             monitor=None)
