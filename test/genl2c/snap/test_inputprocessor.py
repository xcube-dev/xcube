import unittest

from test.sampledata import create_highroc_dataset
from xcube.genl2c.snap.inputprocessor import SnapOlciHighrocL2InputProcessor


class SnapOlciHighrocL2InputProcessorTest(unittest.TestCase):

    def setUp(self):
        self.processor = SnapOlciHighrocL2InputProcessor()

    def test_props(self):
        self.assertEqual('snap-olci-highroc-l2', self.processor.name)
        self.assertEqual('SNAP Sentinel-3 OLCI HIGHROC Level-2 NetCDF inputs', self.processor.description)
        self.assertEqual({'r'}, self.processor.modes)
        self.assertEqual('nc', self.processor.ext)
        self.assertEqual('({expr}) AND !quality_flags.land', self.processor.extra_expr_pattern)

    def test_reprojection_info(self):
        reprojection_info = self.processor.get_reprojection_info(create_highroc_dataset())
        self.assertEqual(('lon', 'lat'), reprojection_info.xy_var_names)
        self.assertEqual(5, reprojection_info.xy_gcp_step)

    def test_read(self):
        with self.assertRaises(OSError):
            self.processor.read('test-nc')

    def _test_pre_process(self):
        # FIXME: this test raises because create_highroc_dataset() does not return compatible SNAP L2 DS.
        ds1 = create_highroc_dataset()
        ds2 = self.processor.pre_process(ds1)
        self.assertIsNot(ds1, ds2)
        # TODO: add more asserts for ds2

    def test_post_process(self):
        ds1 = create_highroc_dataset()
        ds2 = self.processor.post_process(ds1)
        self.assertIsNot(ds1, ds2)
        # TODO: add more asserts for ds2
