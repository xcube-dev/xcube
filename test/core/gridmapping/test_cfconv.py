import unittest

import pyproj

CRS_WGS84 = pyproj.crs.CRS(4326)
CRS_CRS84 = pyproj.crs.CRS.from_string("urn:ogc:def:crs:OGC:1.3:CRS84")
CRS_UTM_33N = pyproj.crs.CRS(32633)

CRS_ROTATED_POLE = pyproj.crs.CRS.from_cf(
    dict(grid_mapping_name="rotated_latitude_longitude",
         grid_north_pole_latitude=32.5,
         grid_north_pole_longitude=170.))


class GetDatasetGridMappingTest(unittest.TestCase):
    def test_it(self):
        pass
