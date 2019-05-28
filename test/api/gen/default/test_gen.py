import os
import unittest
import xarray as xr
import numpy as np

from xcube.api.gen.config import get_config_dict
from xcube.util.dsio import rimraf
from xcube.api.gen.gen import gen_cube
from .helpers import get_inputdata_path


def clean_up():
    files = ['l2c-single.nc', 'l2c.nc', 'l2c.zarr', 'l2c-single.zarr']
    for file in files:
        rimraf(os.path.join('.', file))
        rimraf(os.path.join('.', file + 'temp.nc'))
    rimraf(get_inputdata_path("input.txt"))


class DefaultProcessTest(unittest.TestCase):

    def setUp(self):
        clean_up()

    def tearDown(self):
        clean_up()

    # noinspection PyMethodMayBeStatic
    def test_process_inputs_single(self):
        path, status = process_inputs_wrapper(
            input_files=[get_inputdata_path('20170101-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc')],
            output_name='l2c-single',
            output_writer='netcdf4',
            append_mode=False)
        self.assertEqual(True, status)
        self.assertEqual(os.path.join('.', 'l2c-single.nc'), path)

    def test_process_inputs_append_multiple_nc(self):
        path, status = process_inputs_wrapper(
            input_files=[get_inputdata_path('201701??-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc')],
            output_name='l2c',
            output_writer='netcdf4',
            append_mode=True)
        self.assertEqual(True, status)
        self.assertEqual(os.path.join('.', 'l2c.nc'), path)

    def test_process_inputs_append_multiple_zarr(self):
        path, status = process_inputs_wrapper(
            input_files=[get_inputdata_path('201701??-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc')],
            output_name='l2c',
            output_writer='zarr',
            append_mode=True)
        self.assertEqual(True, status)
        self.assertEqual(os.path.join('.', 'l2c.zarr'), path)

    def test_input_txt(self):
        f = open((os.path.join(os.path.dirname(__file__), 'inputdata', "input.txt")), "w+")
        for i in range(1, 4):
            file_name = "2017010" + str(i) + "-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc"
            file = get_inputdata_path(file_name)
            f.write("%s\n" % file)
        f.close()
        path, status = process_inputs_wrapper(
            input_files=[get_inputdata_path('input.txt')],
            output_name='l2c',
            output_writer='zarr',
            append_mode=True)
        self.assertEqual(True, status)
        self.assertEqual(os.path.join('.', 'l2c.zarr'), path)

    def test_handle_360_lon(self):
        path, status = process_inputs_wrapper(
            input_files=[get_inputdata_path('20170101120000-UKMO-L4_GHRSST-SSTfnd-OSTIAanom-GLOB-v02.0-fv02.0.nc')],
            output_name='l2c-single',
            output_writer='zarr',
            append_mode=True)
        self.assertEqual(True, status)
        self.assertEqual(os.path.join('.', 'l2c-single.zarr'), path)
        lon_360 = _check_output_for_360(os.path.join('.', 'l2c-single.zarr'))
        self.assertEqual(False, lon_360)


# noinspection PyShadowingBuiltins
def process_inputs_wrapper(input_files=None,
                           output_name=None,
                           output_writer='netcdf4',
                           append_mode=False):
    config = get_config_dict(locals())
    return gen_cube(input_processor='default',
                    output_size=(320, 180),
                    output_region=(-4., 47., 12., 56.),
                    output_resampling='Nearest',
                    output_variables=[('analysed_sst', dict(name='SST'))],
                    output_dir='.',
                    dry_run=False,
                    monitor=None,
                    **config)


def _check_output_for_360(output):
    ds = xr.open_zarr(output)
    lon_var = ds.coords['lon']
    lon_size = lon_var.shape[0]
    lon_size_05 = lon_size // 2
    lon_values = lon_var.values
    lon_360 = True
    if not np.any(lon_values[lon_size_05:] > 180.):
        lon_360 = False
    return lon_360
