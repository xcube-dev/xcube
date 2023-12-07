import unittest
import pyproj

from xcube.webapi.ows.coverages.request import CoveragesRequest


class CoveragesRequestTest(unittest.TestCase):
    def test_parse_bbox(self):
        self.assertIsNone(CoveragesRequest({}).bbox)
        self.assertEqual(
            [1.1, 2.2, 3.3, 4.4],
            CoveragesRequest(dict(bbox=['1.1,2.2,3.3,4.4'])).bbox,
        )
        with self.assertRaises(ValueError):
            CoveragesRequest(dict(bbox=['foo,bar,baz']))
        with self.assertRaises(ValueError):
            CoveragesRequest(dict(bbox=['1.1,2.2,3.3']))

    def test_parse_bbox_crs(self):
        self.assertEqual(
            pyproj.CRS('OGC:CRS84'),
            CoveragesRequest({}).bbox_crs,
        )
        self.assertEqual(
            pyproj.CRS(crs_spec := 'EPSG:4326'),
            CoveragesRequest({'bbox-crs': [crs_spec]}).bbox_crs,
        )
        self.assertEqual(
            pyproj.CRS(crs_spec := 'OGC:CRS84'),
            CoveragesRequest({'bbox-crs': [f'[{crs_spec}]']}).bbox_crs,
        )
        with self.assertRaises(ValueError):
            CoveragesRequest({'bbox-crs': ['not a CRS specifier']})

    def test_parse_datetime(self):
        dt0 = '2018-02-12T23:20:52Z'
        dt1 = '2019-02-12T23:20:52Z'
        self.assertIsNone(CoveragesRequest({}).datetime)
        self.assertEqual(dt0, CoveragesRequest({'datetime': [dt0]}).datetime)
        self.assertEqual(
            (dt0, None), CoveragesRequest({'datetime': [f'{dt0}/..']}).datetime
        )
        self.assertEqual(
            (None, dt1), CoveragesRequest({'datetime': [f'../{dt1}']}).datetime
        )
        self.assertEqual(
            (dt0, dt1),
            CoveragesRequest({'datetime': [f'{dt0}/{dt1}']}).datetime,
        )
        with self.assertRaises(ValueError):
            CoveragesRequest({'datetime': [f'{dt0}/{dt0}/{dt1}']})
        with self.assertRaises(ValueError):
            CoveragesRequest({'datetime': ['not a valid time string']})

    def test_parse_subset(self):
        self.assertIsNone(CoveragesRequest({}).subset)
        self.assertEqual(
            dict(Lat=('10', '20'), Lon=('30', None), time='2019-03-27'),
            CoveragesRequest(
                dict(subset=['Lat(10:20),Lon(30:*),time("2019-03-27")'])
            ).subset,
        )
        self.assertEqual(
            dict(
                Lat=(None, '20'), Lon='30', time=('2019-03-27', '2020-03-27')
            ),
            CoveragesRequest(
                dict(
                    subset=[
                        'Lat(*:20),Lon(30),time("2019-03-27":"2020-03-27")'
                    ]
                )
            ).subset,
        )
        with self.assertRaises(ValueError):
            CoveragesRequest({'subset': ['not a valid specifier']})

    def test_parse_subset_crs(self):
        self.assertEqual(
            pyproj.CRS('OGC:CRS84'),
            CoveragesRequest({}).subset_crs,
        )
        self.assertEqual(
            pyproj.CRS(crs_spec := 'EPSG:4326'),
            CoveragesRequest({'subset-crs': [crs_spec]}).subset_crs,
        )
        with self.assertRaises(ValueError):
            CoveragesRequest({'subset-crs': ['not a CRS specifier']})

    def test_parse_properties(self):
        self.assertIsNone(CoveragesRequest({}).properties)
        self.assertEqual(
            ['foo', 'bar', 'baz'],
            CoveragesRequest(dict(properties=['foo,bar,baz'])).properties,
        )

    def test_parse_scale_factor(self):
        self.assertEqual(1, CoveragesRequest({}).scale_factor)
        self.assertEqual(
            1.5, CoveragesRequest({'scale-factor': ['1.5']}).scale_factor
        )
        with self.assertRaises(ValueError):
            CoveragesRequest({'scale-factor': ['this is not a number']})

    def test_parse_scale_axes(self):
        self.assertIsNone(CoveragesRequest({}).scale_axes)
        self.assertEqual(
            dict(Lat=1.5, Lon=2.5),
            CoveragesRequest({'scale-axes': ['Lat(1.5),Lon(2.5)']}).scale_axes
        )
        with self.assertRaises(ValueError):
            CoveragesRequest({'scale-axes': ['Lat(1.5']})
        with self.assertRaises(ValueError):
            CoveragesRequest({'scale-axes': ['Lat(not a number)']})

    def test_parse_scale_size(self):
        self.assertIsNone(CoveragesRequest({}).scale_size)
        self.assertEqual(
            dict(Lat=12.3, Lon=45.6),
            CoveragesRequest({'scale-size': ['Lat(12.3),Lon(45.6)']}).scale_size
        )
        with self.assertRaises(ValueError):
            CoveragesRequest({'scale-size': ['Lat(1.5']})
        with self.assertRaises(ValueError):
            CoveragesRequest({'scale-size': ['Lat(not a number)']})

    def test_parse_crs(self):
        self.assertIsNone(CoveragesRequest({}).crs)
        self.assertEqual(
            pyproj.CRS(crs := 'EPSG:4326'),
            CoveragesRequest({'crs': [crs]}).crs
        )
        with self.assertRaises(ValueError):
            CoveragesRequest({'crs': ['an invalid CRS specifier']})
