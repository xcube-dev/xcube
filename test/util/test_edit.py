import unittest

import yaml
import zarr

from test.sampledata import create_highroc_dataset
from xcube.core.new import new_cube
from xcube.util.chunk import chunk_dataset
from xcube.util.constants import FORMAT_NAME_ZARR
from xcube.util.dsio import rimraf
from xcube.util.edit import edit_metadata
from xcube.util.optimize import optimize_dataset

TEST_CUBE = create_highroc_dataset()

TEST_CUBE_COORDS = chunk_dataset(new_cube(time_periods=3, variables=dict(A=0.5, B=-1.5)),
                                 chunk_sizes=dict(time=1, lat=180, lon=360), format_name=FORMAT_NAME_ZARR)

TEST_CUBE_ZARR = 'test.zarr'

TEST_CUBE_ZARR_COORDS = 'test_coords.zarr'

TEST_CUBE_ZARR_OPTIMIZED = 'test-optimized.zarr'

TEST_CUBE_ZARR_EDIT = 'test_edited_meta.zarr'

TEST_CUBE_ZARR_OPTIMIZED_EDIT = 'test_optimized_edited_meta.zarr'

TEST_NEW_META = {'output_metadata': {'creator_name': 'Brockmann Consult GmbH with love',
                                     'creator_url': 'www.some_very_nice_url.com',
                                     'geospatial_lat_max': 'something around the north pole.'},
                 'conc_chl': {'units': 'happiness'},
                 'some_crazy_var': {'units': 'happiness'}}

TEST_NEW_META_YML = 'test_new_meta.yml'


class EditVariablePropsTest(unittest.TestCase):

    def setUp(self):
        rimraf(TEST_CUBE_ZARR)
        rimraf(TEST_CUBE_ZARR_COORDS)
        rimraf(TEST_NEW_META_YML)
        rimraf(TEST_CUBE_ZARR_EDIT)
        rimraf(TEST_CUBE_ZARR_OPTIMIZED)
        rimraf(TEST_CUBE_ZARR_OPTIMIZED_EDIT)
        TEST_CUBE.to_zarr(TEST_CUBE_ZARR)
        TEST_CUBE_COORDS.to_zarr(TEST_CUBE_ZARR_COORDS)
        with open(TEST_NEW_META_YML, 'w') as outfile:
            yaml.dump(TEST_NEW_META, outfile, default_flow_style=False)

    def tearDown(self):
        rimraf(TEST_CUBE_ZARR)
        rimraf(TEST_CUBE_ZARR_COORDS)
        rimraf(TEST_NEW_META_YML)
        rimraf(TEST_CUBE_ZARR_EDIT)
        rimraf(TEST_CUBE_ZARR_OPTIMIZED)
        rimraf(TEST_CUBE_ZARR_OPTIMIZED_EDIT)

    def test_edit_metadata(self):
        edit_metadata(TEST_CUBE_ZARR, metadata_path=TEST_NEW_META_YML, in_place=False,
                      output_path=TEST_CUBE_ZARR_EDIT, monitor=print)
        ds1 = zarr.open(TEST_CUBE_ZARR)
        ds2 = zarr.open(TEST_CUBE_ZARR_EDIT)
        self.assertEqual(ds1.__len__(), ds2.__len__())
        self.assertEqual(ds1.attrs.__getitem__('start_date'), ds2.attrs.__getitem__('start_date'))
        self.assertEqual('happiness', ds2['conc_chl'].attrs.__getitem__('units'))
        self.assertNotIn('creator_name', ds1.attrs.keys())
        self.assertIn('creator_name', ds2.attrs.keys())

    def test_update_coords_metadata(self):
        edit_metadata(TEST_CUBE_ZARR, metadata_path=TEST_NEW_META_YML, update_coords=True, in_place=False,
                      output_path=TEST_CUBE_ZARR_EDIT, monitor=print)
        ds1 = zarr.open(TEST_CUBE_ZARR)
        ds2 = zarr.open(TEST_CUBE_ZARR_EDIT)
        self.assertEqual(ds1.__len__(), ds2.__len__())
        self.assertEqual(ds1.attrs.__getitem__('start_date'), ds2.attrs.__getitem__('start_date'))
        self.assertEqual(ds1.attrs.__getitem__('start_date'), ds2.attrs.__getitem__('start_date'))
        self.assertEqual('happiness', ds2['conc_chl'].attrs.__getitem__('units'))
        self.assertIn('geospatial_lon_units', ds2.attrs.keys())
        self.assertEqual('degrees_east', ds2.attrs.__getitem__('geospatial_lon_units'))
        self.assertNotIn('geospatial_lat_max', ds1.attrs.keys())
        self.assertNotIn('geospatial_lat_max', ds2.attrs.keys())

    def test_update_coords_metadata_only(self):
        ds1 = zarr.open(TEST_CUBE_ZARR_COORDS)
        delete_list = ['geospatial_lat_max', 'geospatial_lat_min', 'geospatial_lat_units', 'geospatial_lon_max',
                       'geospatial_lon_min', 'geospatial_lon_units', 'time_coverage_end', 'time_coverage_start']
        for attr in ds1.attrs.keys():
            if attr in delete_list:
                ds1.attrs.__delitem__(attr)
        edit_metadata(TEST_CUBE_ZARR_COORDS, update_coords=True, in_place=False,
                      output_path=TEST_CUBE_ZARR_EDIT, monitor=print)
        ds1 = zarr.open(TEST_CUBE_ZARR_COORDS)
        ds2 = zarr.open(TEST_CUBE_ZARR_EDIT)
        for attr in delete_list:
            self.assertNotIn(attr, ds1.attrs.keys())
        self.assertEqual(ds1.__len__(), ds2.__len__())
        self.assertIn('geospatial_lat_max', ds2.attrs.keys())
        self.assertIn('geospatial_lat_min', ds2.attrs.keys())
        self.assertIn('geospatial_lat_resolution', ds2.attrs.keys())
        self.assertIn('geospatial_lat_units', ds2.attrs.keys())
        self.assertIn('geospatial_lon_max', ds2.attrs.keys())
        self.assertEqual(180.0, ds2.attrs.__getitem__('geospatial_lon_max'))
        self.assertEqual(-180.0, ds2.attrs.__getitem__('geospatial_lon_min'))
        self.assertEqual('2010-01-04T00:00:00.000000000', ds2.attrs.__getitem__('time_coverage_end'))
        self.assertEqual('2010-01-01T00:00:00.000000000', ds2.attrs.__getitem__('time_coverage_start'))
        self.assertEqual('degrees_east', ds2.attrs.__getitem__('geospatial_lon_units'))

    def test_edit_metadata_in_place(self):
        edit_metadata(TEST_CUBE_ZARR, metadata_path=TEST_NEW_META_YML, in_place=True, monitor=print)
        ds1 = zarr.open(TEST_CUBE_ZARR)
        self.assertEqual(36, ds1.__len__())
        self.assertEqual('14-APR-2017 10:27:50.183264', ds1.attrs.__getitem__('start_date'))
        self.assertEqual('happiness', ds1['conc_chl'].attrs.__getitem__('units'))
        self.assertIn('creator_name', ds1.attrs.keys())
        self.assertIn('creator_url', ds1.attrs.keys())
        self.assertEqual('Brockmann Consult GmbH with love', ds1.attrs.__getitem__('creator_name'))
        self.assertEqual('www.some_very_nice_url.com', ds1.attrs.__getitem__('creator_url'))

    def test_edit_zmetadata(self):
        optimize_dataset(TEST_CUBE_ZARR, unchunk_coords=True, output_path=TEST_CUBE_ZARR_OPTIMIZED)
        edit_metadata(TEST_CUBE_ZARR_OPTIMIZED, metadata_path=TEST_NEW_META_YML, in_place=False,
                      output_path=TEST_CUBE_ZARR_OPTIMIZED_EDIT, monitor=print)
        ds1 = zarr.open(TEST_CUBE_ZARR)
        ds2 = zarr.convenience.open_consolidated(TEST_CUBE_ZARR_OPTIMIZED_EDIT)
        self.assertEqual(ds1.__len__(), ds2.__len__())
        self.assertEqual(ds1.attrs.__getitem__('start_date'), ds2.attrs.__getitem__('start_date'))
        self.assertEqual('happiness', ds2['conc_chl'].attrs.__getitem__('units'))
        self.assertNotIn('creator_name', ds1.attrs.keys())
        self.assertIn('creator_name', ds2.attrs.keys())

    def test_failures(self):
        with self.assertRaises(RuntimeError) as cm:
            edit_metadata('pippo', in_place=True, exception_type=RuntimeError, metadata_path=TEST_NEW_META_YML)
        self.assertEqual('Input path must point to ZARR dataset directory.', f'{cm.exception}')

        with self.assertRaises(RuntimeError) as cm:
            edit_metadata(TEST_CUBE_ZARR, exception_type=RuntimeError, metadata_path=TEST_NEW_META_YML)
        self.assertEqual('Output path must be given.', f'{cm.exception}')

        with self.assertRaises(RuntimeError) as cm:
            edit_metadata(TEST_CUBE_ZARR, output_path=TEST_CUBE_ZARR, exception_type=RuntimeError,
                          metadata_path=TEST_NEW_META_YML)
        self.assertEqual('Output path already exists.', f'{cm.exception}')

        with self.assertRaises(RuntimeError) as cm:
            edit_metadata(TEST_CUBE_ZARR, output_path='./' + TEST_CUBE_ZARR, exception_type=RuntimeError,
                          metadata_path=TEST_NEW_META_YML)
        self.assertEqual('Output path already exists.', f'{cm.exception}')
