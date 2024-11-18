# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest
import xcube.core.new
import xcube.webapi.ows.coverages.util as util


class UtilTest(unittest.TestCase):
    def setUp(self):
        self.ds_latlon = xcube.core.new.new_cube()
        self.ds_xy = xcube.core.new.new_cube(x_name="x", y_name="y")

    def test_get_h_dim(self):
        self.assertEqual("lon", util.get_h_dim(self.ds_latlon))
        self.assertEqual("x", util.get_h_dim(self.ds_xy))

    def test_get_v_dim(self):
        self.assertEqual("lat", util.get_v_dim(self.ds_latlon))
        self.assertEqual("y", util.get_v_dim(self.ds_xy))
