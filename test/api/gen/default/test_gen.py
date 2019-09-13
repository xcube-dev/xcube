import os
import unittest
from typing import Tuple, Optional

import numpy as np
import xarray as xr

from xcube.api.gen.config import get_config_dict
from xcube.api.gen.gen import gen_cube
from xcube.util.dsio import rimraf
from .helpers import get_inputdata_path


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
        status, output = gen_cube_wrapper(
            [get_inputdata_path('20170101-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc')], 'l2c-single.nc')
        self.assertEqual(True, status)
        self.assertTrue('\nstep 8 of 8: creating input slice in l2c-single.nc...\n' in output)

    def test_process_inputs_append_multiple_nc(self):
        status, output = gen_cube_wrapper(
            [get_inputdata_path('201701??-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc')], 'l2c.nc',
            sort_mode=True)
        self.assertEqual(True, status)
        self.assertTrue('\nstep 8 of 8: creating input slice in l2c.nc...\n' in output)
        self.assertTrue('\nstep 8 of 8: appending input slice to l2c.nc...\n' in output)

    def test_process_inputs_append_multiple_zarr(self):
        status, output = gen_cube_wrapper(
            [get_inputdata_path('201701??-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc')], 'l2c.zarr',
            sort_mode=True)
        self.assertEqual(True, status)
        self.assertTrue('\nstep 8 of 8: creating input slice in l2c.zarr...\n' in output)
        self.assertTrue('\nstep 8 of 8: appending input slice to l2c.zarr...\n' in output)

    def test_process_inputs_insert_multiple_zarr(self):
        status, output = gen_cube_wrapper(
            [get_inputdata_path('20170102-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc'),
             get_inputdata_path('20170103-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc'),
             get_inputdata_path('20170101-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc')], 'l2c.zarr',
            sort_mode=False)
        self.assertEqual(True, status)
        self.assertTrue('\nstep 8 of 8: creating input slice in l2c.zarr...\n' in output)
        self.assertTrue('\nstep 8 of 8: appending input slice to l2c.zarr...\n' in output)
        self.assertTrue('\nstep 8 of 8: inserting input slice before index 0 in l2c.zarr...\n' in output)

    def test_process_inputs_replace_multiple_zarr(self):
        status, output = gen_cube_wrapper(
            [get_inputdata_path('20170101-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc'),
             get_inputdata_path('20170102-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc'),
             get_inputdata_path('20170103-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc'),
             get_inputdata_path('20170102-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc')], 'l2c.zarr',
            sort_mode=False)
        self.assertEqual(True, status)
        self.assertTrue('\nstep 8 of 8: creating input slice in l2c.zarr...\n' in output)
        self.assertTrue('\nstep 8 of 8: appending input slice to l2c.zarr...\n' in output)
        self.assertTrue('\nstep 8 of 8: replacing input slice at index 1 in l2c.zarr...\n' in output)

    def test_input_txt(self):
        f = open((os.path.join(os.path.dirname(__file__), 'inputdata', "input.txt")), "w+")
        for i in range(1, 4):
            file_name = "2017010" + str(i) + "-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc"
            file = get_inputdata_path(file_name)
            f.write("%s\n" % file)
        f.close()
        status, output = gen_cube_wrapper([get_inputdata_path('input.txt')], 'l2c.zarr', sort_mode=True)
        self.assertEqual(True, status)

    def test_handle_360_lon(self):
        status, output = gen_cube_wrapper(
            [get_inputdata_path('20170101120000-UKMO-L4_GHRSST-SSTfnd-OSTIAanom-GLOB-v02.0-fv02.0.nc')],
            'l2c-single.zarr', sort_mode=True)
        self.assertEqual(True, status)
        ds = xr.open_zarr('l2c-single.zarr')
        self.assertIn('lon', ds.coords)
        self.assertFalse(np.any(ds.coords['lon'] > 180.))

    def test_illegal_proc(self):
        with self.assertRaises(ValueError) as e:
            gen_cube_wrapper(
                [get_inputdata_path('20170101120000-UKMO-L4_GHRSST-SSTfnd-OSTIAanom-GLOB-v02.0-fv02.0.nc')],
                'l2c-single.zarr', sort_mode=True, input_processor_name=None)
        self.assertEqual('Missing input_processor_name', f'{e.exception}')

        with self.assertRaises(ValueError) as e:
            gen_cube_wrapper(
                [get_inputdata_path('20170101120000-UKMO-L4_GHRSST-SSTfnd-OSTIAanom-GLOB-v02.0-fv02.0.nc')],
                'l2c-single.zarr', sort_mode=True, input_processor_name='chris-proba')
        self.assertEqual("Unknown input_processor_name 'chris-proba'", f'{e.exception}')


# noinspection PyShadowingBuiltins
def gen_cube_wrapper(input_paths, output_path, sort_mode=False, input_processor_name='default') \
        -> Tuple[bool, Optional[str]]:
    output = None

    def output_monitor(msg):
        nonlocal output
        if output is None:
            output = msg + '\n'
        else:
            output += msg + '\n'

    config = get_config_dict(dict(input_paths=input_paths, output_path=output_path))
    return gen_cube(input_processor_name=input_processor_name,
                    output_size=(320, 180),
                    output_region=(-4., 47., 12., 56.),
                    output_resampling='Nearest',
                    output_variables=[('analysed_sst', dict(name='SST'))],
                    sort_mode=sort_mode,
                    dry_run=False,
                    monitor=output_monitor,
                    **config), output
