import os.path
import os.path
import unittest

from xcube.api import new_cube
from xcube.api.readwrite import write_cube
from xcube.util.dsio import rimraf
from xcube.webapi.s3util import list_bucket, list_bucket_result_to_xml


class ListBucketTest(unittest.TestCase):
    S3_BUCKET = os.path.join(os.path.dirname(__file__), "s3-bucket")
    TEST_CUBE_1 = os.path.join(S3_BUCKET, "test-1.zarr")
    TEST_CUBE_2 = os.path.join(S3_BUCKET, "test-2.zarr")

    def test_list_bucket(self):
        self.maxDiff = None
        list_bucket_result = list_bucket(self.S3_BUCKET)
        self.assertListBucketResult(list_bucket_result)
        self.assertIsInstance(list_bucket_result.get('Contents'), list)
        self.assertEqual(170, len(list_bucket_result.get('Contents')))
        self.assertNotIn('CommonPrefixes', list_bucket_result)

    def test_list_bucket_truncated(self):
        self.maxDiff = None
        list_bucket_result = list_bucket(self.S3_BUCKET, max_keys=5, last_modified='?')
        self.assertListBucketResult(list_bucket_result, max_keys=5, is_truncated=True,
                                    next_marker='test-2.zarr/time_bnds/')
        self.assertIsInstance(list_bucket_result.get('Contents'), list)
        self.assertEqual([{'ETag': '"9b4c896c64396373b28d5f4c5c438b3f"',
                           'Key': 'test-1.zarr/',
                           'LastModified': '?',
                           'Size': 0,
                           'StorageClass': 'STANDARD'},
                          {'ETag': '"e861244d88ada176417ca9cef653c252"',
                           'Key': 'test-1.zarr/.zattrs',
                           'LastModified': '?',
                           'Size': 426,
                           'StorageClass': 'STANDARD'},
                          {'ETag': '"8181702529c9172a386fd239a538fb8c"',
                           'Key': 'test-1.zarr/.zgroup',
                           'LastModified': '?',
                           'Size': 24,
                           'StorageClass': 'STANDARD'},
                          {'ETag': '"3f48face6f2b707cf5c8c15dd2074e6f"',
                           'Key': 'test-1.zarr/lat/',
                           'LastModified': '?',
                           'Size': 0,
                           'StorageClass': 'STANDARD'},
                          {'ETag': '"1fd08df28f426c8cdfbbf0ddc53ae659"',
                           'Key': 'test-1.zarr/lat/.zarray',
                           'LastModified': '?',
                           'Size': 317,
                           'StorageClass': 'STANDARD'}],
                         list_bucket_result.get('Contents'))
        self.assertNotIn('CommonPrefixes', list_bucket_result)

    def test_list_bucket_prefix(self):
        self.maxDiff = None
        list_bucket_result = list_bucket(self.S3_BUCKET, prefix='test-1.zarr/')
        self.assertListBucketResult(list_bucket_result, prefix='test-1.zarr/')
        self.assertIsInstance(list_bucket_result.get('Contents'), list)
        self.assertEqual(85, len(list_bucket_result.get('Contents')))
        self.assertNotIn('CommonPrefixes', list_bucket_result)

    def test_list_bucket_delimiter(self):
        self.maxDiff = None
        list_bucket_result = list_bucket(self.S3_BUCKET, delimiter='/')
        self.assertListBucketResult(list_bucket_result, delimiter='/')
        self.assertNotIn('Contents', list_bucket_result)
        self.assertIsInstance(list_bucket_result.get('CommonPrefixes'), list)
        self.assertEqual(['test-1.zarr/',
                          'test-2.zarr/'],
                         list_bucket_result.get('CommonPrefixes'))

    def test_list_bucket_delimiter_prefix(self):
        self.maxDiff = None
        list_bucket_result = list_bucket(self.S3_BUCKET, delimiter='/', prefix='test-2.zarr/', last_modified='?')
        self.assertListBucketResult(list_bucket_result, delimiter='/', prefix='test-2.zarr/')
        self.assertIsInstance(list_bucket_result.get('Contents'), list)
        self.assertIsInstance(list_bucket_result.get('CommonPrefixes'), list)
        self.assertEqual([{'ETag': '"d19efdd038f8a3dc35b4dd03bb280a4a"',
                           'Key': 'test-2.zarr/',
                           'LastModified': '?',
                           'Size': 0,
                           'StorageClass': 'STANDARD'},
                          {'ETag': '"258058e6205449b0931127c9a46e7907"',
                           'Key': 'test-2.zarr/.zattrs',
                           'LastModified': '?',
                           'Size': 426,
                           'StorageClass': 'STANDARD'},
                          {'ETag': '"eee5470de5d094747fb51087fa617b9c"',
                           'Key': 'test-2.zarr/.zgroup',
                           'LastModified': '?',
                           'Size': 24,
                           'StorageClass': 'STANDARD'}],
                         list_bucket_result.get('Contents'))
        self.assertEqual(['test-2.zarr/lat/',
                          'test-2.zarr/lat_bnds/',
                          'test-2.zarr/lon/',
                          'test-2.zarr/lon_bnds/',
                          'test-2.zarr/precipitation/',
                          'test-2.zarr/temperature/',
                          'test-2.zarr/time/',
                          'test-2.zarr/time_bnds/'],
                         list_bucket_result.get('CommonPrefixes'))

    def test_list_bucket_delimiter_prefix_2(self):
        self.maxDiff = None
        list_bucket_result = list_bucket(self.S3_BUCKET, delimiter='/', prefix='test-2.zarr', last_modified='?')
        self.assertListBucketResult(list_bucket_result, delimiter='/', prefix='test-2.zarr')
        self.assertNotIn('Contents', list_bucket_result)
        self.assertIsInstance(list_bucket_result.get('CommonPrefixes'), list)
        self.assertEqual(['test-2.zarr/'],
                         list_bucket_result.get('CommonPrefixes'))

    def assertListBucketResult(self, list_bucket_result, name="s3-bucket", prefix=None,
                               delimiter=None, max_keys=1000, is_truncated=False, marker=None, next_marker=None):
        self.assertIsInstance(list_bucket_result, dict)
        self.assertEqual(name, list_bucket_result.get('Name'))
        self.assertEqual(prefix, list_bucket_result.get('Prefix'))
        self.assertEqual(delimiter, list_bucket_result.get('Delimiter'))
        self.assertEqual(max_keys, list_bucket_result.get('MaxKeys'))
        self.assertEqual(is_truncated, list_bucket_result.get('IsTruncated'))
        self.assertEqual(marker, list_bucket_result.get('Marker'))
        self.assertEqual(next_marker, list_bucket_result.get('NextMarker'))

    def test_list_bucket_result_to_xml(self):
        self.maxDiff = None
        list_bucket_result = list_bucket(self.S3_BUCKET, delimiter='/', max_keys=10, prefix='test-1.zarr/',
                                         last_modified='2019-06-24 20:43:40.862072')
        xml = list_bucket_result_to_xml(list_bucket_result)
        with open(os.path.join(os.path.dirname(__file__), 's3', 'list-bucket-result-1.xml')) as fp:
            expected_xml = fp.read()
        self.assertEqual(expected_xml, xml)

    @classmethod
    def setUpClass(cls) -> None:
        rimraf(cls.S3_BUCKET)
        os.mkdir(cls.S3_BUCKET)
        cube = new_cube(time_periods=3,
                        variables=dict(precipitation=0.9,
                                       temperature=278.3)).chunk(dict(time=1, lat=90, lon=90))
        write_cube(cube, cls.TEST_CUBE_1, "zarr", cube_asserted=True)
        write_cube(cube, cls.TEST_CUBE_2, "zarr", cube_asserted=True)

    @classmethod
    def tearDownClass(cls) -> None:
        rimraf(cls.S3_BUCKET)
