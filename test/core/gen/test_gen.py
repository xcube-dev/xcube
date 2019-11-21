import os
import unittest
from typing import Any, Dict, Optional, Tuple

import numpy as np
import xarray as xr

from test.core.gen.helpers import get_inputdata_path
from xcube.core.dsio import rimraf
from xcube.core.gen.config import get_config_dict
from xcube.core.gen.gen import gen_cube


def clean_up():
    files = ['l2c-single.nc', 'l2c-single.zarr', 'l2c.nc', 'l2c.zarr']
    for file in files:
        rimraf(file)
        rimraf(file + '.temp.nc')  # May remain from Netcdf4DatasetIO.append()
    rimraf(get_inputdata_path("input.txt"))


class DefaultProcessTest(unittest.TestCase):

    def setUp(self):
        clean_up()

    def tearDown(self):
        clean_up()

    def test_process_inputs_single_nc(self):
        status, output = gen_cube_wrapper(
            [get_inputdata_path('20170101-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc')], 'l2c-single.nc')
        self.assertEqual(True, status)
        self.assertTrue('\nstep 8 of 8: creating input slice in l2c-single.nc...\n' in output)
        self.assert_cube_ok(xr.open_dataset('l2c-single.nc', autoclose=True), 1,
                            dict(date_modified=None,
                                 time_coverage_start='2016-12-31T12:00:00.000000000',
                                 time_coverage_end='2017-01-01T12:00:00.000000000'))

    def test_process_inputs_append_multiple_nc(self):
        status, output = gen_cube_wrapper(
            [get_inputdata_path('201701??-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc')], 'l2c.nc',
            no_sort_mode=False)
        self.assertEqual(True, status)
        self.assertTrue('\nstep 8 of 8: creating input slice in l2c.nc...\n' in output)
        self.assertTrue('\nstep 8 of 8: appending input slice to l2c.nc...\n' in output)
        self.assert_cube_ok(xr.open_dataset('l2c.nc', autoclose=True), 3,
                            dict(date_modified=None,
                                 time_coverage_start='2016-12-31T12:00:00.000000000',
                                 time_coverage_end='2017-01-03T12:00:00.000000000'))

    def test_process_inputs_single_zarr(self):
        status, output = gen_cube_wrapper(
            [get_inputdata_path('20170101-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc')], 'l2c-single.zarr')
        self.assertEqual(True, status)
        self.assertTrue('\nstep 8 of 8: creating input slice in l2c-single.zarr...\n' in output)
        self.assert_cube_ok(xr.open_zarr('l2c-single.zarr'), 1,
                            dict(date_modified=None,
                                 time_coverage_start='2016-12-31T12:00:00.000000000',
                                 time_coverage_end='2017-01-01T12:00:00.000000000'))

    def test_process_inputs_append_multiple_zarr(self):
        status, output = gen_cube_wrapper(
            [get_inputdata_path('201701??-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc')], 'l2c.zarr',
            no_sort_mode=False)
        self.assertEqual(True, status)
        self.assertTrue('\nstep 8 of 8: creating input slice in l2c.zarr...\n' in output)
        self.assertTrue('\nstep 8 of 8: appending input slice to l2c.zarr...\n' in output)
        self.assert_cube_ok(xr.open_zarr('l2c.zarr'), 3,
                            dict(date_modified=None,
                                 time_coverage_start='2016-12-31T12:00:00.000000000',
                                 time_coverage_end='2017-01-03T12:00:00.000000000'))

    def test_process_inputs_insert_multiple_zarr(self):
        status, output = gen_cube_wrapper(
            [get_inputdata_path('20170102-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc'),
             get_inputdata_path('20170103-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc'),
             get_inputdata_path('20170101-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc')], 'l2c.zarr',
            no_sort_mode=True)
        self.assertEqual(True, status)
        self.assertTrue('\nstep 8 of 8: creating input slice in l2c.zarr...\n' in output)
        self.assertTrue('\nstep 8 of 8: appending input slice to l2c.zarr...\n' in output)
        self.assertTrue('\nstep 8 of 8: inserting input slice before index 0 in l2c.zarr...\n' in output)
        self.assert_cube_ok(xr.open_zarr('l2c.zarr'), 3,
                            dict(date_modified=None,
                                 time_coverage_start='2016-12-31T12:00:00.000000000',
                                 time_coverage_end='2017-01-03T12:00:00.000000000'))

    def test_process_inputs_replace_multiple_zarr(self):
        status, output = gen_cube_wrapper(
            [get_inputdata_path('20170101-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc'),
             get_inputdata_path('20170102-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc'),
             get_inputdata_path('20170103-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc'),
             get_inputdata_path('20170102-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc')], 'l2c.zarr',
            no_sort_mode=True)
        self.assertEqual(True, status)
        self.assertTrue('\nstep 8 of 8: creating input slice in l2c.zarr...\n' in output)
        self.assertTrue('\nstep 8 of 8: appending input slice to l2c.zarr...\n' in output)
        self.assertTrue('\nstep 8 of 8: replacing input slice at index 1 in l2c.zarr...\n' in output)
        self.assert_cube_ok(xr.open_zarr('l2c.zarr'), 3,
                            dict(date_modified=None,
                                 time_coverage_start='2016-12-31T12:00:00.000000000',
                                 time_coverage_end='2017-01-03T12:00:00.000000000'))

    def test_input_txt(self):
        f = open((os.path.join(os.path.dirname(__file__), 'inputdata', "input.txt")), "w+")
        for i in range(1, 4):
            file_name = f"2017010{i}-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc"
            file = get_inputdata_path(file_name)
            f.write("%s\n" % file)
        f.close()
        status, output = gen_cube_wrapper([get_inputdata_path('input.txt')], 'l2c.zarr', no_sort_mode=False)
        self.assertEqual(True, status)
        self.assert_cube_ok(xr.open_zarr('l2c.zarr'), 3,
                            dict(time_coverage_start='2016-12-31T12:00:00.000000000',
                                 time_coverage_end='2017-01-03T12:00:00.000000000'))

    def assert_cube_ok(self, cube: xr.Dataset, expected_time_dim: int, expected_extra_attrs: Dict[str, Any]):
        self.assertEqual({'lat': 180, 'lon': 320, 'bnds': 2, 'time': expected_time_dim}, cube.dims)
        self.assertEqual({'lon', 'lat', 'time', 'lon_bnds', 'lat_bnds', 'time_bnds'}, set(cube.coords))
        self.assertEqual({'analysed_sst'}, set(cube.data_vars))
        expected_attrs = dict(title='Test Cube',
                              project='xcube',
                              date_modified=None,
                              geospatial_lon_min=-4.0,
                              geospatial_lon_max=12.0,
                              geospatial_lon_resolution=0.05,
                              geospatial_lon_units='degrees_east',
                              geospatial_lat_min=47.0,
                              geospatial_lat_max=56.0,
                              geospatial_lat_resolution=0.05,
                              geospatial_lat_units='degrees_north')
        expected_attrs.update(expected_extra_attrs)
        for k, v in expected_attrs.items():
            self.assertIn(k, cube.attrs)
            if v is not None:
                self.assertEqual(v, cube.attrs[k], msg=f'key {k!r}')

    def test_handle_360_lon(self):
        status, output = gen_cube_wrapper(
            [get_inputdata_path('20170101120000-UKMO-L4_GHRSST-SSTfnd-OSTIAanom-GLOB-v02.0-fv02.0.nc')],
            'l2c-single.zarr', no_sort_mode=False)
        self.assertEqual(True, status)
        ds = xr.open_zarr('l2c-single.zarr')
        self.assertIn('lon', ds.coords)
        self.assertFalse(np.any(ds.coords['lon'] > 180.))

    def test_illegal_proc(self):
        with self.assertRaises(ValueError) as e:
            gen_cube_wrapper(
                [get_inputdata_path('20170101120000-UKMO-L4_GHRSST-SSTfnd-OSTIAanom-GLOB-v02.0-fv02.0.nc')],
                'l2c-single.zarr', no_sort_mode=False, input_processor_name="")
        self.assertEqual('input_processor_name must not be empty', f'{e.exception}')

        with self.assertRaises(ValueError) as e:
            gen_cube_wrapper(
                [get_inputdata_path('20170101120000-UKMO-L4_GHRSST-SSTfnd-OSTIAanom-GLOB-v02.0-fv02.0.nc')],
                'l2c-single.zarr', no_sort_mode=False, input_processor_name='chris-proba')
        self.assertEqual("Unknown input_processor_name 'chris-proba'", f'{e.exception}')


# noinspection PyShadowingBuiltins
def gen_cube_wrapper(input_paths, output_path, no_sort_mode=False, input_processor_name=None) \
        -> Tuple[bool, Optional[str]]:
    output = None

    def output_monitor(msg):
        nonlocal output
        if output is None:
            output = msg + '\n'
        else:
            output += msg + '\n'

    config = get_config_dict(
        input_paths=input_paths,
        input_processor_name=input_processor_name,
        output_path=output_path,
        output_size='320,180',
        output_region='-4,47,12,56',
        output_resampling='Nearest',
        output_variables='analysed_sst',
        no_sort_mode=no_sort_mode,
    )

    output_metadata = dict(
        title='Test Cube',
        project='xcube',
    )

    return gen_cube(dry_run=False, monitor=output_monitor, output_metadata=output_metadata, **config), output
