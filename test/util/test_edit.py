import unittest

import yaml
import zarr

from test.sampledata import create_highroc_dataset
from xcube.util.dsio import rimraf
from xcube.util.edit import edit_metadata

TEST_CUBE = create_highroc_dataset()

TEST_CUBE_ZARR = 'test.zarr'

TEST_CUBE_ZARR_EDIT = 'test_edited_meta.zarr'

TEST_NEW_META = {'output_metadata': {'creator_name': 'Brockmann Consult GmbH with love',
                                     'creator_url': 'www.some_very_nice_url.com'},
                 'conc_chl': {'units': 'happiness'},
                 'some_crazy_var': {'units': 'happiness'}}

TEST_NEW_META_YML = 'test_new_meta.yml'


class EditVariablePropsTest(unittest.TestCase):

    def setUp(self):
        rimraf(TEST_CUBE_ZARR)
        rimraf(TEST_NEW_META_YML)
        rimraf(TEST_CUBE_ZARR_EDIT)
        TEST_CUBE.to_zarr(TEST_CUBE_ZARR)
        with open(TEST_NEW_META_YML, 'w') as outfile:
            yaml.dump(TEST_NEW_META, outfile, default_flow_style=False)

    def tearDown(self):
        rimraf(TEST_CUBE_ZARR)
        rimraf(TEST_NEW_META_YML)
        rimraf(TEST_CUBE_ZARR_EDIT)

    def test_edit_metadata_partly(self):
        edit_metadata(TEST_CUBE_ZARR, metadata_path=TEST_NEW_META_YML, in_place=False,
                      output_path=TEST_CUBE_ZARR_EDIT)
        ds1 = zarr.open(TEST_CUBE_ZARR)
        ds2 = zarr.open(TEST_CUBE_ZARR_EDIT)
        self.assertEqual(ds1.__len__(), ds2.__len__())
        self.assertEqual(ds1.attrs.__getitem__('start_date'), ds2.attrs.__getitem__('start_date'))
        self.assertEqual('happiness', ds2['conc_chl'].attrs.__getitem__('units'))
        self.assertNotIn('creator_name', ds1.attrs.keys())
        self.assertIn('creator_name', ds2.attrs.keys())

    def test_edit_metadata_partly_in_place(self):
        edit_metadata(TEST_CUBE_ZARR, metadata_path=TEST_NEW_META_YML, in_place=True)
        ds1 = zarr.open(TEST_CUBE_ZARR)
        self.assertEqual(36, ds1.__len__())
        self.assertEqual('14-APR-2017 10:27:50.183264', ds1.attrs.__getitem__('start_date'))
        self.assertEqual('happiness', ds1['conc_chl'].attrs.__getitem__('units'))
        self.assertIn('creator_name', ds1.attrs.keys())
        self.assertIn('creator_url', ds1.attrs.keys())
        self.assertEqual('Brockmann Consult GmbH with love', ds1.attrs.__getitem__('creator_name'))
        self.assertEqual('www.some_very_nice_url.com', ds1.attrs.__getitem__('creator_url'))