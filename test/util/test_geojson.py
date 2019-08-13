import unittest

from xcube.util.geojson import GeoJSON


class GeoJSONTest(unittest.TestCase):

    def test_is_point(self):
        self.assertTrue(GeoJSON.is_point(dict(type='Point', coordinates=[2.13, 42.2])))
        self.assertFalse(GeoJSON.is_point(dict(type='Feature', properties=None)))

    def test_is_geometry(self):
        self.assertTrue(GeoJSON.is_geometry(dict(type='Point', coordinates=[2.13, 42.2])))
        self.assertTrue(GeoJSON.is_geometry(dict(type='Point', coordinates=None)))
        self.assertFalse(GeoJSON.is_geometry(dict(type='Point')))

        self.assertTrue(GeoJSON.is_geometry(dict(type='GeometryCollection', geometries=None)))
        self.assertTrue(GeoJSON.is_geometry(dict(type='GeometryCollection', geometries=[])))
        self.assertFalse(GeoJSON.is_geometry(dict(type='GeometryCollection')))

        self.assertFalse(GeoJSON.is_geometry(dict(type='Feature', properties=None)))

    def test_is_feature(self):
        self.assertTrue(GeoJSON.is_feature(dict(type='Feature',
                                                geometry=dict(type='Point',
                                                              coordinates=[2.13, 42.2]))))
        self.assertFalse(GeoJSON.is_feature(dict(type='Point', coordinates=[2.13, 42.2])))

    def test_is_feature_collection(self):
        self.assertTrue(GeoJSON.is_feature_collection(dict(type='FeatureCollection',
                                                           features=[dict(type='Feature',
                                                                          geometry=dict(type='Point',
                                                                                        coordinates=[
                                                                                            2.13,
                                                                                            42.2]))])))
        self.assertFalse(GeoJSON.is_feature_collection(dict(type='Point', coordinates=[2.13, 42.2])))

    def test_get_type_name(self):
        self.assertEqual('Feature',
                         GeoJSON.get_type_name(dict(type='Feature')))
        self.assertEqual(None,
                         GeoJSON.get_type_name(dict(pype='Feature')))
        self.assertEqual(None,
                         GeoJSON.get_type_name(dict()))
        self.assertEqual(None,
                         GeoJSON.get_type_name(17))

    def test_get_feature_geometry(self):
        self.assertEqual(dict(type='Point', coordinates=[2.13, 42.2]),
                         GeoJSON.get_feature_geometry(dict(type='Feature',
                                                           geometry=dict(type='Point', coordinates=[2.13, 42.2]))))
        self.assertEqual(None,
                         GeoJSON.get_feature_geometry(dict(type='Pleature',
                                                           geometry=dict(type='Point', coordinates=[2.13, 42.2]))))
        self.assertEqual(None,
                         GeoJSON.get_feature_geometry(dict(type='Feature',
                                                           geometry=dict(type='Point'))))
        self.assertEqual(None,
                         GeoJSON.get_feature_geometry(dict(type='Feature',
                                                           geometry=17)))
