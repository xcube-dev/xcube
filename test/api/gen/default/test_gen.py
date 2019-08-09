import os
import unittest

import numpy as np
import xarray as xr

from xcube.api.gen.config import get_config_dict
from xcube.api.gen.gen import gen_cube
from xcube.util.dsio import rimraf
from .helpers import get_inputdata_path, create_input_txt


def clean_up():
    files = ['l2c-single.nc', 'l2c.nc', 'l2c.zarr', 'l2c-single.zarr']
    for file in files:
        rimraf(file)
        rimraf(file + '.temp.nc')  # May remain from Netcdf4DatasetIO.append()
    rimraf(get_inputdata_path("input.txt"))


class DefaultProcessTest(unittest.TestCase):

    def setUp(self):
        clean_up()

    def tearDown(self):
        clean_up()

    # noinspection PyMethodMayBeStatic
    def test_process_inputs_single(self):
        status = gen_cube_wrapper(
            [get_inputdata_path('20170101-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc')],
            'l2c-single.nc',
            False
        )
        self.assertEqual(True, status)

    def test_process_inputs_append_multiple_nc(self):
        status = gen_cube_wrapper(
            [get_inputdata_path('201701??-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc')],
            'l2c.nc',
            True
        )
        self.assertEqual(True, status)

    def test_process_inputs_append_multiple_zarr(self):
        status = gen_cube_wrapper(
            [get_inputdata_path('201701??-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc')],
            'l2c.zarr',
            True
        )
        self.assertEqual(True, status)

    def test_input_txt(self):
        create_input_txt(range(1, 4))
        status = gen_cube_wrapper(
            [get_inputdata_path('input.txt')],
            'l2c.zarr',
            True
        )
        self.assertEqual(True, status)

    def test_handle_360_lon(self):
        status = gen_cube_wrapper(
            [get_inputdata_path('20170101120000-UKMO-L4_GHRSST-SSTfnd-OSTIAanom-GLOB-v02.0-fv02.0.nc')],
            'l2c-single.zarr',
            True
        )
        self.assertEqual(True, status)
        ds = xr.open_zarr('l2c-single.zarr')
        self.assertIn('lon', ds.coords)
        self.assertFalse(np.any(ds.coords['lon'] > 180.))

    def test_update_metadata_with_cubegen_param(self):
        create_input_txt(range(1, 4))
        status = gen_cube_wrapper(
            [get_inputdata_path('input.txt')],
            'l2c.zarr',
            True
        )
        self.assertEqual(True, status)
        ds = xr.open_zarr('l2c.zarr')
        self.assertIn('history', ds.attrs.keys())


# noinspection PyShadowingBuiltins
def gen_cube_wrapper(input_paths, output_path, append_mode):
    config = get_config_dict(locals())
    return gen_cube(input_processor='default',
                    output_size=(320, 180),
                    output_region=(-4., 47., 12., 56.),
                    output_resampling='Nearest',
                    output_variables=[('analysed_sst', dict(name='SST'))],
                    dry_run=False,
                    monitor=None,
                    **config)
