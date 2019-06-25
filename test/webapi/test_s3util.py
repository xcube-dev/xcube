import os.path
import os.path
import unittest

from xcube.api import new_cube
from xcube.api.readwrite import write_cube
from xcube.util.dsio import rimraf
from xcube.webapi.s3util import list_bucket, list_bucket_result_to_xml


class BucketTest(unittest.TestCase):
    S3_BUCKET = os.path.join(os.path.dirname(__file__), "s3-bucket")
    TEST_CUBE_1 = os.path.join(S3_BUCKET, "test-1.zarr")
    TEST_CUBE_2 = os.path.join(S3_BUCKET, "test-2.zarr")

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


class ListBucketTest(BucketTest):

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
        self.assertEqual([{'ETag': '"b25cd90af1027bca40aac125cd677632"',
                           'Key': 'test-1.zarr/',
                           'LastModified': '?',
                           'Size': 0,
                           'StorageClass': 'STANDARD'},
                          {'ETag': '"63c85aa57274ab7eda56a0d095df193b"',
                           'Key': 'test-1.zarr/.zattrs',
                           'LastModified': '?',
                           'Size': 426,
                           'StorageClass': 'STANDARD'},
                          {'ETag': '"3e4069dee4d3a2e5c8214197389d9532"',
                           'Key': 'test-1.zarr/.zgroup',
                           'LastModified': '?',
                           'Size': 24,
                           'StorageClass': 'STANDARD'},
                          {'ETag': '"faac4687801157f63477ba6d3d7dab54"',
                           'Key': 'test-1.zarr/lat/',
                           'LastModified': '?',
                           'Size': 0,
                           'StorageClass': 'STANDARD'},
                          {'ETag': '"82616df189209368fae5bd8caaf0f209"',
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
        self.assertEqual([{'ETag': '"2c510152490933efdd58ab4c8a7f811c"',
                           'Key': 'test-2.zarr/',
                           'LastModified': '?',
                           'Size': 0,
                           'StorageClass': 'STANDARD'},
                          {'ETag': '"5d8946b3d4ab4cfb9f000d9458a2f38b"',
                           'Key': 'test-2.zarr/.zattrs',
                           'LastModified': '?',
                           'Size': 426,
                           'StorageClass': 'STANDARD'},
                          {'ETag': '"818d43163d18ab128774c57fa346b4d3"',
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

    def test_list_bucket_result_to_xml(self):
        self.maxDiff = None
        list_bucket_result = list_bucket(self.S3_BUCKET, delimiter='/', max_keys=10, prefix='test-1.zarr/',
                                         last_modified='2019-06-24 20:43:40.862072')
        self.assertListBucketResult(list_bucket_result, delimiter='/', max_keys=10, prefix='test-1.zarr/')
        xml = list_bucket_result_to_xml(list_bucket_result)
        with open(os.path.join(os.path.dirname(__file__), 's3', 'list-bucket-result-1.xml')) as fp:
            expected_xml = fp.read()
        self.assertEqual(expected_xml, xml)

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
