import yaml
import zarr

from test.cli.helpers import CliDataTest
from test.core.test_edit import TEST_CUBE, TEST_CUBE_ZARR, TEST_CUBE_ZARR_EDIT, TEST_NEW_META, TEST_NEW_META_YML, \
    TEST_CUBE_ZARR_COORDS, TEST_CUBE_COORDS
from xcube.core.dsio import rimraf

TEST_CUBE_ZARR_EDIT_DEFAULT = 'test-edited.zarr'


class EditMetadaDataTest(CliDataTest):
    def _clear_outputs(self):
        rimraf(TEST_CUBE_ZARR)
        rimraf(TEST_CUBE_ZARR_EDIT)
        rimraf(TEST_NEW_META_YML)
        rimraf(TEST_CUBE_ZARR_EDIT_DEFAULT)
        rimraf(TEST_CUBE_ZARR_COORDS)

    def setUp(self):
        self._clear_outputs()
        TEST_CUBE.to_zarr(TEST_CUBE_ZARR)
        TEST_CUBE_COORDS.to_zarr(TEST_CUBE_ZARR_COORDS)
        with open(TEST_NEW_META_YML, 'w') as outfile:
            yaml.dump(TEST_NEW_META, outfile, default_flow_style=False)

    def tearDown(self):
        self._clear_outputs()

    def test_defaults(self):
        result = self.invoke_cli(['edit', TEST_CUBE_ZARR, '-M', TEST_NEW_META_YML])
        self.assertEqual(0, result.exit_code)

        ds1 = zarr.open(TEST_CUBE_ZARR)
        ds2 = zarr.open(TEST_CUBE_ZARR_EDIT_DEFAULT)
        self.assertEqual(ds1.__len__(), ds2.__len__())
        self.assertEqual(ds1.attrs.__getitem__('start_date'), ds2.attrs.__getitem__('start_date'))
        self.assertEqual('happiness', ds2['conc_chl'].attrs.__getitem__('units'))
        self.assertNotIn('creator_name', ds1.attrs.keys())
        self.assertIn('creator_name', ds2.attrs.keys())

    def test_user_output(self):
        result = self.invoke_cli(['edit', TEST_CUBE_ZARR, '-M', TEST_NEW_META_YML, '-o', TEST_CUBE_ZARR_EDIT])
        self.assertEqual(0, result.exit_code)

        ds1 = zarr.open(TEST_CUBE_ZARR)
        ds2 = zarr.open(TEST_CUBE_ZARR_EDIT)
        self.assertEqual(ds1.__len__(), ds2.__len__())
        self.assertEqual(ds1.attrs.__getitem__('start_date'), ds2.attrs.__getitem__('start_date'))
        self.assertEqual('happiness', ds2['conc_chl'].attrs.__getitem__('units'))
        self.assertNotIn('creator_name', ds1.attrs.keys())
        self.assertIn('creator_name', ds2.attrs.keys())

    def test_in_place(self):
        result = self.invoke_cli(['edit', '-I', TEST_CUBE_ZARR, '-M', TEST_NEW_META_YML])
        self.assertEqual(0, result.exit_code)

        ds1 = zarr.open(TEST_CUBE_ZARR)
        self.assertEqual(36, ds1.__len__())
        self.assertEqual('14-APR-2017 10:27:50.183264', ds1.attrs.__getitem__('start_date'))
        self.assertEqual('happiness', ds1['conc_chl'].attrs.__getitem__('units'))
        self.assertIn('creator_name', ds1.attrs.keys())
        self.assertIn('creator_url', ds1.attrs.keys())
        self.assertEqual('Brockmann Consult GmbH with love', ds1.attrs.__getitem__('creator_name'))
        self.assertEqual('www.some_very_nice_url.com', ds1.attrs.__getitem__('creator_url'))

    def test_update_coords_only(self):
        ds1 = zarr.open(TEST_CUBE_ZARR_COORDS)
        delete_list = ['geospatial_lat_max', 'geospatial_lat_min', 'geospatial_lat_units', 'geospatial_lon_max',
                       'geospatial_lon_min', 'geospatial_lon_units', 'time_coverage_end', 'time_coverage_start']
        for attr in ds1.attrs.keys():
            if attr in delete_list:
                ds1.attrs.__delitem__(attr)
        result = self.invoke_cli(['edit', TEST_CUBE_ZARR_COORDS, '-o', TEST_CUBE_ZARR_EDIT, '-C'])
        self.assertEqual(0, result.exit_code)

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


class EditTest(CliDataTest):

    def test_help_option(self):
        result = self.invoke_cli(['edit', '--help'])
        self.assertEqual(0, result.exit_code)
