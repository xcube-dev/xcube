import os.path
import os.path
import unittest
from abc import abstractmethod, ABCMeta
from typing import Dict

from xcube.api import new_cube
from xcube.api.readwrite import write_cube
from xcube.util.dsio import rimraf
from xcube.webapi.s3util import list_bucket_result_to_xml, list_bucket_keys, list_bucket_v2, list_bucket_v1


class ListBucketTest(unittest.TestCase, metaclass=ABCMeta):
    S3_BUCKET = os.path.join(os.path.dirname(__file__), "s3-bucket")
    TEST_CUBE_1 = os.path.join(S3_BUCKET, "test-1.zarr")
    TEST_CUBE_2 = os.path.join(S3_BUCKET, "test-2.zarr")

    BUCKET_DICT = {'bibo.zarr': TEST_CUBE_1,
                   'bert.zarr': TEST_CUBE_2}

    def setUp(self):
        self.maxDiff = None

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

    @abstractmethod
    def list_bucket(self, bucket_dict, **kwargs):
        pass

    @abstractmethod
    def assert_list_bucket_result(self, list_bucket_result, **kwargs):
        pass


# noinspection PyUnresolvedReferences
class ListBucketV12TestsMixin:

    def test_list_bucket_keys(self):
        result = [key_path for key_path in list_bucket_keys(self.BUCKET_DICT)]
        self.assertEqual(EXPECTED_KEYS, [key for key, _ in result])
        self.assertEqual(170, len(EXPECTED_KEYS))

    def test_list_bucket_v12(self):
        list_bucket_result = self.list_bucket(self.BUCKET_DICT)
        self.assert_list_bucket_result(list_bucket_result)
        self.assertIsInstance(list_bucket_result.get('Contents'), list)
        self.assertEqual(170, len(list_bucket_result.get('Contents')))
        self.assertNotIn('CommonPrefixes', list_bucket_result)

    def test_list_bucket_v12_prefix(self):
        list_bucket_result = self.list_bucket(self.BUCKET_DICT, prefix='bibo.zarr/')
        self.assert_list_bucket_result(list_bucket_result, prefix='bibo.zarr/')
        self.assertIsInstance(list_bucket_result.get('Contents'), list)
        self.assertEqual(85, len(list_bucket_result.get('Contents')))
        self.assertNotIn('CommonPrefixes', list_bucket_result)

    def test_list_bucket_v12_delimiter(self):
        list_bucket_result = self.list_bucket(self.BUCKET_DICT, delimiter='/')
        self.assert_list_bucket_result(list_bucket_result, delimiter='/')
        self.assertNotIn('Contents', list_bucket_result)
        self.assertIsInstance(list_bucket_result.get('CommonPrefixes'), list)
        self.assertEqual([{'Prefix': 'bert.zarr/'},
                          {'Prefix': 'bibo.zarr/'}],
                         list_bucket_result.get('CommonPrefixes'))

    def test_list_bucket_v12_delimiter_prefix(self):
        list_bucket_result = self.list_bucket(self.BUCKET_DICT, delimiter='/',
                                              prefix='bert.zarr/', last_modified='?')
        self.assert_list_bucket_result(list_bucket_result, delimiter='/',
                                       prefix='bert.zarr/')
        self.assertIsInstance(list_bucket_result.get('Contents'), list)
        self.assertIsInstance(list_bucket_result.get('CommonPrefixes'), list)
        self.assertEqual([{'ETag': '"2c510152490933efdd58ab4c8a7f811c"',
                           'Key': 'bert.zarr/',
                           'LastModified': '?',
                           'Size': 0,
                           'StorageClass': 'STANDARD'},
                          {'ETag': '"5d8946b3d4ab4cfb9f000d9458a2f38b"',
                           'Key': 'bert.zarr/.zattrs',
                           'LastModified': '?',
                           'Size': 426,
                           'StorageClass': 'STANDARD'},
                          {'ETag': '"818d43163d18ab128774c57fa346b4d3"',
                           'Key': 'bert.zarr/.zgroup',
                           'LastModified': '?',
                           'Size': 24,
                           'StorageClass': 'STANDARD'}],
                         list_bucket_result.get('Contents'))
        self.assertEqual([{'Prefix': 'bert.zarr/lat/'},
                          {'Prefix': 'bert.zarr/lat_bnds/'},
                          {'Prefix': 'bert.zarr/lon/'},
                          {'Prefix': 'bert.zarr/lon_bnds/'},
                          {'Prefix': 'bert.zarr/precipitation/'},
                          {'Prefix': 'bert.zarr/temperature/'},
                          {'Prefix': 'bert.zarr/time/'},
                          {'Prefix': 'bert.zarr/time_bnds/'}],
                         list_bucket_result.get('CommonPrefixes'))

    def test_list_bucket_v12_delimiter_prefix_2(self):
        list_bucket_result = self.list_bucket(self.BUCKET_DICT, delimiter='/',
                                              prefix='bert.zarr', last_modified='?')
        self.assert_list_bucket_result(list_bucket_result, delimiter='/',
                                       prefix='bert.zarr')
        self.assertNotIn('Contents', list_bucket_result)
        self.assertIsInstance(list_bucket_result.get('CommonPrefixes'), list)
        self.assertEqual([{'Prefix': 'bert.zarr/'}],
                         list_bucket_result.get('CommonPrefixes'))


class ListBucketV1Test(ListBucketTest, ListBucketV12TestsMixin):
    def test_list_bucket_v1_truncated(self):
        list_bucket_result = self.list_bucket(self.BUCKET_DICT,
                                              max_keys=5, last_modified='?')
        self.assert_list_bucket_result(list_bucket_result, max_keys=5, is_truncated=True,
                                       next_marker='bert.zarr/lat/.zattrs')
        self.assertIsInstance(list_bucket_result.get('Contents'), list)
        self.assertEqual([{'ETag': '"2c510152490933efdd58ab4c8a7f811c"',
                           'Key': 'bert.zarr/',
                           'LastModified': '?',
                           'Size': 0,
                           'StorageClass': 'STANDARD'},
                          {'ETag': '"5d8946b3d4ab4cfb9f000d9458a2f38b"',
                           'Key': 'bert.zarr/.zattrs',
                           'LastModified': '?',
                           'Size': 426,
                           'StorageClass': 'STANDARD'},
                          {'ETag': '"818d43163d18ab128774c57fa346b4d3"',
                           'Key': 'bert.zarr/.zgroup',
                           'LastModified': '?',
                           'Size': 24,
                           'StorageClass': 'STANDARD'},
                          {'ETag': '"1f42995f4d5156dc9efe3d97f5df3cfe"',
                           'Key': 'bert.zarr/lat/',
                           'LastModified': '?',
                           'Size': 0,
                           'StorageClass': 'STANDARD'},
                          {'ETag': '"f95c0565c0cdb4bec1132434755abf5a"',
                           'Key': 'bert.zarr/lat/.zarray',
                           'LastModified': '?',
                           'Size': 317,
                           'StorageClass': 'STANDARD'}],
                         list_bucket_result.get('Contents'))
        self.assertNotIn('CommonPrefixes', list_bucket_result)

    def test_list_bucket_v1_result_to_xml(self):
        list_bucket_result = self.list_bucket(self.BUCKET_DICT, delimiter='/',
                                              max_keys=10, prefix='bibo.zarr/',
                                              last_modified='2019-06-24 20:43:40.862072')
        self.assert_list_bucket_result(list_bucket_result, delimiter='/',
                                       max_keys=10, prefix='bibo.zarr/')
        xml = list_bucket_result_to_xml(list_bucket_result)
        with open(os.path.join(os.path.dirname(__file__), 's3', 'list-bucket-v1-result.xml')) as fp:
            expected_xml = fp.read()
        self.assertEqual(expected_xml, xml)

    def list_bucket(self, bucket_entries: Dict[str, str], **kwargs):
        return list_bucket_v1(bucket_entries, **kwargs)

    def assert_list_bucket_result(self,
                                  list_bucket_result,
                                  name="s3bucket",
                                  prefix=None,
                                  delimiter=None,
                                  max_keys=1000,
                                  is_truncated=False,
                                  marker=None,
                                  next_marker=None):
        self.assertIsInstance(list_bucket_result, dict)
        self.assertEqual(name, list_bucket_result.get('Name'))
        self.assertEqual(prefix, list_bucket_result.get('Prefix'))
        self.assertEqual(delimiter, list_bucket_result.get('Delimiter'))
        self.assertEqual(max_keys, list_bucket_result.get('MaxKeys'))
        self.assertEqual(is_truncated, list_bucket_result.get('IsTruncated'))
        self.assertEqual(marker, list_bucket_result.get('Marker'))
        self.assertEqual(next_marker, list_bucket_result.get('NextMarker'))


class ListBucketV2Test(ListBucketTest, ListBucketV12TestsMixin):

    def test_list_bucket_v2_truncated(self):
        list_bucket_result = self.list_bucket(self.BUCKET_DICT,
                                              max_keys=5, last_modified='?')
        self.assert_list_bucket_result(list_bucket_result, max_keys=5, is_truncated=True,
                                       next_continuation_token='bert.zarr/lat/.zattrs')
        self.assertIsInstance(list_bucket_result.get('Contents'), list)
        self.assertEqual([{'ETag': '"2c510152490933efdd58ab4c8a7f811c"',
                           'Key': 'bert.zarr/',
                           'LastModified': '?',
                           'Size': 0,
                           'StorageClass': 'STANDARD'},
                          {'ETag': '"5d8946b3d4ab4cfb9f000d9458a2f38b"',
                           'Key': 'bert.zarr/.zattrs',
                           'LastModified': '?',
                           'Size': 426,
                           'StorageClass': 'STANDARD'},
                          {'ETag': '"818d43163d18ab128774c57fa346b4d3"',
                           'Key': 'bert.zarr/.zgroup',
                           'LastModified': '?',
                           'Size': 24,
                           'StorageClass': 'STANDARD'},
                          {'ETag': '"1f42995f4d5156dc9efe3d97f5df3cfe"',
                           'Key': 'bert.zarr/lat/',
                           'LastModified': '?',
                           'Size': 0,
                           'StorageClass': 'STANDARD'},
                          {'ETag': '"f95c0565c0cdb4bec1132434755abf5a"',
                           'Key': 'bert.zarr/lat/.zarray',
                           'LastModified': '?',
                           'Size': 317,
                           'StorageClass': 'STANDARD'}],
                         list_bucket_result.get('Contents'))
        self.assertNotIn('CommonPrefixes', list_bucket_result)

    def test_list_bucket_v2_result_to_xml(self):
        list_bucket_result = self.list_bucket(self.BUCKET_DICT, delimiter='/',
                                              max_keys=10, prefix='bibo.zarr/',
                                              last_modified='2019-06-24 20:43:40.862072')
        self.assert_list_bucket_result(list_bucket_result, delimiter='/',
                                       max_keys=10, prefix='bibo.zarr/')
        xml = list_bucket_result_to_xml(list_bucket_result)
        with open(os.path.join(os.path.dirname(__file__), 's3', 'list-bucket-v2-result.xml')) as fp:
            expected_xml = fp.read()
        self.assertEqual(expected_xml, xml)

    def list_bucket(self, bucket_entries: Dict[str, str], **kwargs):
        return list_bucket_v2(bucket_entries, **kwargs)

    def assert_list_bucket_result(self,
                                  list_bucket_result,
                                  name="s3bucket",
                                  prefix=None,
                                  delimiter=None, max_keys=1000,
                                  is_truncated=False,
                                  start_after=None,
                                  continuation_token=None,
                                  next_continuation_token=None):
        self.assertIsInstance(list_bucket_result, dict)
        self.assertEqual(name, list_bucket_result.get('Name'))
        self.assertEqual(prefix, list_bucket_result.get('Prefix'))
        self.assertEqual(delimiter, list_bucket_result.get('Delimiter'))
        self.assertEqual(max_keys, list_bucket_result.get('MaxKeys'))
        self.assertEqual(is_truncated, list_bucket_result.get('IsTruncated'))
        self.assertEqual(start_after, list_bucket_result.get('Marker'))
        self.assertEqual(continuation_token, list_bucket_result.get('ContinuationToken'))
        self.assertEqual(next_continuation_token, list_bucket_result.get('NextContinuationToken'))


EXPECTED_KEYS = [
    'bert.zarr/',
    'bert.zarr/.zattrs',
    'bert.zarr/.zgroup',
    'bert.zarr/lat/',
    'bert.zarr/lat/.zarray',
    'bert.zarr/lat/.zattrs',
    'bert.zarr/lat/0',
    'bert.zarr/lat_bnds/',
    'bert.zarr/lat_bnds/.zarray',
    'bert.zarr/lat_bnds/.zattrs',
    'bert.zarr/lat_bnds/0.0',
    'bert.zarr/lat_bnds/1.0',
    'bert.zarr/lon/',
    'bert.zarr/lon/.zarray',
    'bert.zarr/lon/.zattrs',
    'bert.zarr/lon/0',
    'bert.zarr/lon_bnds/',
    'bert.zarr/lon_bnds/.zarray',
    'bert.zarr/lon_bnds/.zattrs',
    'bert.zarr/lon_bnds/0.0',
    'bert.zarr/lon_bnds/1.0',
    'bert.zarr/lon_bnds/2.0',
    'bert.zarr/lon_bnds/3.0',
    'bert.zarr/precipitation/',
    'bert.zarr/precipitation/.zarray',
    'bert.zarr/precipitation/.zattrs',
    'bert.zarr/precipitation/0.0.0',
    'bert.zarr/precipitation/0.0.1',
    'bert.zarr/precipitation/0.0.2',
    'bert.zarr/precipitation/0.0.3',
    'bert.zarr/precipitation/0.1.0',
    'bert.zarr/precipitation/0.1.1',
    'bert.zarr/precipitation/0.1.2',
    'bert.zarr/precipitation/0.1.3',
    'bert.zarr/precipitation/1.0.0',
    'bert.zarr/precipitation/1.0.1',
    'bert.zarr/precipitation/1.0.2',
    'bert.zarr/precipitation/1.0.3',
    'bert.zarr/precipitation/1.1.0',
    'bert.zarr/precipitation/1.1.1',
    'bert.zarr/precipitation/1.1.2',
    'bert.zarr/precipitation/1.1.3',
    'bert.zarr/precipitation/2.0.0',
    'bert.zarr/precipitation/2.0.1',
    'bert.zarr/precipitation/2.0.2',
    'bert.zarr/precipitation/2.0.3',
    'bert.zarr/precipitation/2.1.0',
    'bert.zarr/precipitation/2.1.1',
    'bert.zarr/precipitation/2.1.2',
    'bert.zarr/precipitation/2.1.3',
    'bert.zarr/temperature/',
    'bert.zarr/temperature/.zarray',
    'bert.zarr/temperature/.zattrs',
    'bert.zarr/temperature/0.0.0',
    'bert.zarr/temperature/0.0.1',
    'bert.zarr/temperature/0.0.2',
    'bert.zarr/temperature/0.0.3',
    'bert.zarr/temperature/0.1.0',
    'bert.zarr/temperature/0.1.1',
    'bert.zarr/temperature/0.1.2',
    'bert.zarr/temperature/0.1.3',
    'bert.zarr/temperature/1.0.0',
    'bert.zarr/temperature/1.0.1',
    'bert.zarr/temperature/1.0.2',
    'bert.zarr/temperature/1.0.3',
    'bert.zarr/temperature/1.1.0',
    'bert.zarr/temperature/1.1.1',
    'bert.zarr/temperature/1.1.2',
    'bert.zarr/temperature/1.1.3',
    'bert.zarr/temperature/2.0.0',
    'bert.zarr/temperature/2.0.1',
    'bert.zarr/temperature/2.0.2',
    'bert.zarr/temperature/2.0.3',
    'bert.zarr/temperature/2.1.0',
    'bert.zarr/temperature/2.1.1',
    'bert.zarr/temperature/2.1.2',
    'bert.zarr/temperature/2.1.3',
    'bert.zarr/time/',
    'bert.zarr/time/.zarray',
    'bert.zarr/time/.zattrs',
    'bert.zarr/time/0',
    'bert.zarr/time_bnds/',
    'bert.zarr/time_bnds/.zarray',
    'bert.zarr/time_bnds/.zattrs',
    'bert.zarr/time_bnds/0.0',
    'bibo.zarr/',
    'bibo.zarr/.zattrs',
    'bibo.zarr/.zgroup',
    'bibo.zarr/lat/',
    'bibo.zarr/lat/.zarray',
    'bibo.zarr/lat/.zattrs',
    'bibo.zarr/lat/0',
    'bibo.zarr/lat_bnds/',
    'bibo.zarr/lat_bnds/.zarray',
    'bibo.zarr/lat_bnds/.zattrs',
    'bibo.zarr/lat_bnds/0.0',
    'bibo.zarr/lat_bnds/1.0',
    'bibo.zarr/lon/',
    'bibo.zarr/lon/.zarray',
    'bibo.zarr/lon/.zattrs',
    'bibo.zarr/lon/0',
    'bibo.zarr/lon_bnds/',
    'bibo.zarr/lon_bnds/.zarray',
    'bibo.zarr/lon_bnds/.zattrs',
    'bibo.zarr/lon_bnds/0.0',
    'bibo.zarr/lon_bnds/1.0',
    'bibo.zarr/lon_bnds/2.0',
    'bibo.zarr/lon_bnds/3.0',
    'bibo.zarr/precipitation/',
    'bibo.zarr/precipitation/.zarray',
    'bibo.zarr/precipitation/.zattrs',
    'bibo.zarr/precipitation/0.0.0',
    'bibo.zarr/precipitation/0.0.1',
    'bibo.zarr/precipitation/0.0.2',
    'bibo.zarr/precipitation/0.0.3',
    'bibo.zarr/precipitation/0.1.0',
    'bibo.zarr/precipitation/0.1.1',
    'bibo.zarr/precipitation/0.1.2',
    'bibo.zarr/precipitation/0.1.3',
    'bibo.zarr/precipitation/1.0.0',
    'bibo.zarr/precipitation/1.0.1',
    'bibo.zarr/precipitation/1.0.2',
    'bibo.zarr/precipitation/1.0.3',
    'bibo.zarr/precipitation/1.1.0',
    'bibo.zarr/precipitation/1.1.1',
    'bibo.zarr/precipitation/1.1.2',
    'bibo.zarr/precipitation/1.1.3',
    'bibo.zarr/precipitation/2.0.0',
    'bibo.zarr/precipitation/2.0.1',
    'bibo.zarr/precipitation/2.0.2',
    'bibo.zarr/precipitation/2.0.3',
    'bibo.zarr/precipitation/2.1.0',
    'bibo.zarr/precipitation/2.1.1',
    'bibo.zarr/precipitation/2.1.2',
    'bibo.zarr/precipitation/2.1.3',
    'bibo.zarr/temperature/',
    'bibo.zarr/temperature/.zarray',
    'bibo.zarr/temperature/.zattrs',
    'bibo.zarr/temperature/0.0.0',
    'bibo.zarr/temperature/0.0.1',
    'bibo.zarr/temperature/0.0.2',
    'bibo.zarr/temperature/0.0.3',
    'bibo.zarr/temperature/0.1.0',
    'bibo.zarr/temperature/0.1.1',
    'bibo.zarr/temperature/0.1.2',
    'bibo.zarr/temperature/0.1.3',
    'bibo.zarr/temperature/1.0.0',
    'bibo.zarr/temperature/1.0.1',
    'bibo.zarr/temperature/1.0.2',
    'bibo.zarr/temperature/1.0.3',
    'bibo.zarr/temperature/1.1.0',
    'bibo.zarr/temperature/1.1.1',
    'bibo.zarr/temperature/1.1.2',
    'bibo.zarr/temperature/1.1.3',
    'bibo.zarr/temperature/2.0.0',
    'bibo.zarr/temperature/2.0.1',
    'bibo.zarr/temperature/2.0.2',
    'bibo.zarr/temperature/2.0.3',
    'bibo.zarr/temperature/2.1.0',
    'bibo.zarr/temperature/2.1.1',
    'bibo.zarr/temperature/2.1.2',
    'bibo.zarr/temperature/2.1.3',
    'bibo.zarr/time/',
    'bibo.zarr/time/.zarray',
    'bibo.zarr/time/.zattrs',
    'bibo.zarr/time/0',
    'bibo.zarr/time_bnds/',
    'bibo.zarr/time_bnds/.zarray',
    'bibo.zarr/time_bnds/.zattrs',
    'bibo.zarr/time_bnds/0.0'
]
