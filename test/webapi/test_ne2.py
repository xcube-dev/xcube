from unittest import TestCase

from xcube.webapi import ne2


class NaturalEarth2Test(TestCase):
    def test_natural_earth_2_pyramid(self):
        pyramid = ne2.NaturalEarth2Image.get_pyramid()

        tile = pyramid.get_tile(0, 0, 0)
        self.assertIsNotNone(tile)
        self.assertEqual(12067, len(tile))

        tile = pyramid.get_tile(7, 3, 2)
        self.assertIsNotNone(tile)
        self.assertEqual(9032, len(tile))
