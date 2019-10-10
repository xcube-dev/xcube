import yaml
import zarr

from test.cli.test_cli import CliDataTest
from test.util.test_edit import TEST_CUBE, TEST_CUBE_ZARR, TEST_CUBE_ZARR_EDIT, TEST_NEW_META, TEST_NEW_META_YML
from xcube.util.dsio import rimraf

TEST_CUBE_ZARR_EDIT_DEFAULT = 'test-edited.zarr'


class EditMetadaDataTest(CliDataTest):
    def _clear_outputs(self):
        rimraf(TEST_CUBE_ZARR)
        rimraf(TEST_CUBE_ZARR_EDIT)
        rimraf(TEST_NEW_META_YML)
        rimraf(TEST_CUBE_ZARR_EDIT_DEFAULT)

    def setUp(self):
        self._clear_outputs()
        TEST_CUBE.to_zarr(TEST_CUBE_ZARR)
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


class EditTest(CliDataTest):

    def test_help_option(self):
        result = self.invoke_cli(['edit', '--help'])
        self.assertEqual(0, result.exit_code)
