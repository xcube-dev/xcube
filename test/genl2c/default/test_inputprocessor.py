import unittest

from xcube.genl2c.default.inputprocessor import DefaultInputProcessor


class DefaultInputProcessorTest(unittest.TestCase):

    def setUp(self):
        self.processor = DefaultInputProcessor()

    def test_props(self):
        # TODO by forman: implement me
        pass
        # self.assertEqual('rbins-seviri-highroc-scene-l2', self.processor.name)
        # self.assertEqual('RBINS SEVIRI HIGHROC single-scene Level-2 NetCDF inputs', self.processor.description)
        # self.assertEqual('netcdf4', self.processor.input_reader)

    def test_reprojection_info(self):
        # TODO by forman: implement me
        pass
        # reprojection_info = self.processor.get_reprojection_info(create_rbins_seviri_scene_dataset())
        # self.assertEqual(('lon', 'lat'), reprojection_info.xy_var_names)
        # self.assertEqual(1, reprojection_info.xy_gcp_step)

    def test_pre_process(self):
        # TODO by forman: implement me
        pass
        # ds1 = create_rbins_seviri_scene_dataset()
        # ds2 = self.processor.pre_process(ds1)
        # self.assertIs(ds1, ds2)

    def test_post_process(self):
        # TODO by forman: implement me
        pass
        # ds1 = create_rbins_seviri_scene_dataset()
        # ds2 = self.processor.post_process(ds1)
        # self.assertIs(ds1, ds2)
