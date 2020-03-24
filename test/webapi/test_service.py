import re
import unittest

from xcube.webapi.service import url_pattern, new_default_config


class DefaultConfigTest(unittest.TestCase):
    def test_new_default_config(self):
        config = new_default_config(["/home/bibo/data/cube-1.zarr",
                                     "/home/bibo/data/cube-2.nc"],
                                    dict(conc_chl=(0.0, 20.0),
                                         conc_tsm=(0.0, 12.0, 'plasma')))
        self.assertEqual({
            'Datasets': [
                {
                    'FileSystem': 'local',
                    'Format': 'zarr',
                    'Identifier': 'dataset_1',
                    'Path': '/home/bibo/data/cube-1.zarr',
                    'Title': 'Dataset #1'
                },
                {
                    'FileSystem': 'local',
                    'Format': 'netcdf4',
                    'Identifier': 'dataset_2',
                    'Path': '/home/bibo/data/cube-2.nc',
                    'Title': 'Dataset #2'
                }
            ],
            'Styles': [
                {'Identifier': 'default',
                 'ColorMappings': {
                     'conc_chl': {'ValueRange': [0.0, 20.0]},
                     'conc_tsm': {'ColorBar': 'plasma',
                                  'ValueRange': [0.0, 12.0]}},
                 }
            ]},
            config)

        with self.assertRaises(ValueError) as cm:
            new_default_config(["/home/bibo/data/cube-1.zarr",
                                "/home/bibo/data/cube-2.nc"],
                               dict(conc_chl=20.0,
                                    conc_tsm=(0.0, 12.0, 'plasma')))
        self.assertEqual("illegal style: conc_chl=20.0", f"{cm.exception}")


class UrlPatternTest(unittest.TestCase):
    def test_url_pattern_works(self):
        re_pattern = url_pattern('/open/{{id1}}ws/{{id2}}wf')
        matcher = re.fullmatch(re_pattern, '/open/34ws/a66wf')
        self.assertIsNotNone(matcher)
        self.assertEqual(matcher.groupdict(), {'id1': '34', 'id2': 'a66'})

        re_pattern = url_pattern('/open/ws{{id1}}/wf{{id2}}')
        matcher = re.fullmatch(re_pattern, '/open/ws34/wfa66')
        self.assertIsNotNone(matcher)
        self.assertEqual(matcher.groupdict(), {'id1': '34', 'id2': 'a66'})

        re_pattern = url_pattern('/datasets/{{ds_id}}/data.zarr/(?P<path>.*)')

        matcher = re.fullmatch(re_pattern, '/datasets/S2PLUS_2017/data.zarr/')
        self.assertIsNotNone(matcher)
        self.assertEqual(matcher.groupdict(), {'ds_id': 'S2PLUS_2017', 'path': ''})

        matcher = re.fullmatch(re_pattern, '/datasets/S2PLUS_2017/data.zarr/conc_chl/.zattrs')
        self.assertIsNotNone(matcher)
        self.assertEqual(matcher.groupdict(), {'ds_id': 'S2PLUS_2017', 'path': 'conc_chl/.zattrs'})

        x = 'C%3A%5CUsers%5CNorman%5CIdeaProjects%5Cccitools%5Cect-core%5Ctest%5Cui%5CTEST_WS_3'
        re_pattern = url_pattern('/ws/{{base_dir}}/res/{{res_name}}/add')
        matcher = re.fullmatch(re_pattern, '/ws/%s/res/SST/add' % x)
        self.assertIsNotNone(matcher)
        self.assertEqual(matcher.groupdict(), {'base_dir': x, 'res_name': 'SST'})

    def test_url_pattern_ok(self):
        self.assertEqual('/version',
                         url_pattern('/version'))
        self.assertEqual(r'(?P<num>[^\;\/\?\:\@\&\=\+\$\,]+)/get',
                         url_pattern('{{num}}/get'))
        self.assertEqual(r'/open/(?P<ws_name>[^\;\/\?\:\@\&\=\+\$\,]+)',
                         url_pattern('/open/{{ws_name}}'))
        self.assertEqual(r'/open/ws(?P<id1>[^\;\/\?\:\@\&\=\+\$\,]+)/wf(?P<id2>[^\;\/\?\:\@\&\=\+\$\,]+)',
                         url_pattern('/open/ws{{id1}}/wf{{id2}}'))
        self.assertEqual(r'/datasets/(?P<ds_id>[^\;\/\?\:\@\&\=\+\$\,]+)/data.zip/(.*)',
                         url_pattern('/datasets/{{ds_id}}/data.zip/(.*)'))

    def test_url_pattern_fail(self):
        with self.assertRaises(ValueError) as cm:
            url_pattern('/open/{{ws/name}}')
        self.assertEqual(str(cm.exception), 'name in {{name}} must be a valid identifier, but got "ws/name"')

        with self.assertRaises(ValueError) as cm:
            url_pattern('/info/{{id}')
        self.assertEqual(str(cm.exception), 'no matching "}}" after "{{" in "/info/{{id}"')
