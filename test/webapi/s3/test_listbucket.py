# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os.path
import os.path
import unittest
from abc import abstractmethod, ABCMeta

from xcube.core.new import new_cube
from xcube.webapi.s3.listbucket import list_bucket_result_to_xml
from xcube.webapi.s3.listbucket import list_s3_bucket_v1
from xcube.webapi.s3.listbucket import list_s3_bucket_v2
from xcube.webapi.s3.objectstorage import ObjectStorage


class S3BucketTest(unittest.TestCase, metaclass=ABCMeta):

    def setUp(self):
        self.maxDiff = None

        cube = new_cube(time_periods=3,
                        variables=dict(precipitation=0.9,
                                       temperature=278.3)).chunk(
            dict(time=1, lat=90, lon=90))

        self.object_storage = ObjectStorage({
            'bibo.zarr': cube,
            'bert.zarr': cube.copy(),
        })


class ListS3BucketTest(S3BucketTest, metaclass=ABCMeta):

    @abstractmethod
    def list_bucket(self, bucket_dict, **kwargs):
        pass

    @abstractmethod
    def assert_list_bucket_result(self, list_bucket_result, **kwargs):
        pass


# noinspection PyUnresolvedReferences
class ListS3BucketV12TestsMixin:

    def test_list_bucket_v12(self):
        list_bucket_result = self.list_bucket()
        self.assert_list_bucket_result(list_bucket_result)
        self.assertIsInstance(list_bucket_result.get('Contents'), list)
        self.assertEqual(158, len(list_bucket_result.get('Contents')))
        self.assertNotIn('CommonPrefixes', list_bucket_result)

    def test_list_bucket_v12_prefix(self):
        list_bucket_result = self.list_bucket(prefix='bibo.zarr/')
        self.assert_list_bucket_result(list_bucket_result,
                                       prefix='bibo.zarr/')
        self.assertIsInstance(list_bucket_result.get('Contents'), list)
        self.assertEqual(79, len(list_bucket_result.get('Contents')))
        self.assertNotIn('CommonPrefixes', list_bucket_result)

    def test_list_bucket_v12_delimiter(self):
        list_bucket_result = self.list_bucket(delimiter='/')
        self.assert_list_bucket_result(list_bucket_result,
                                       delimiter='/')
        self.assertNotIn('Contents', list_bucket_result)
        self.assertIsInstance(list_bucket_result.get('CommonPrefixes'), list)
        self.assertEqual([{'Prefix': 'bert.zarr/'},
                          {'Prefix': 'bibo.zarr/'}],
                         list_bucket_result.get('CommonPrefixes'))

    def test_list_bucket_v12_delimiter_prefix(self):
        list_bucket_result = self.list_bucket(delimiter='/',
                                              prefix='bert.zarr/')
        self.assert_list_bucket_result(list_bucket_result,
                                       delimiter='/',
                                       prefix='bert.zarr/')
        self.assertIsInstance(list_bucket_result.get('Contents'), list)
        self.assertIsInstance(list_bucket_result.get('CommonPrefixes'), list)
        self.assertEqual(
            [
                {'ETag': '"16a49349c15b9cd60c5d8a05fc6ad649"',
                 'Key': 'bert.zarr/.zattrs',
                 'LastModified': '2019-06-24T20:43:40.862Z',
                 'Size': -1,
                 'StorageClass': 'STANDARD'},
                {'ETag': '"0a44cbe4de5e2936112efc2ed25b6223"',
                 'Key': 'bert.zarr/.zgroup',
                 'LastModified': '2019-06-24T20:43:40.862Z',
                 'Size': -1,
                 'StorageClass': 'STANDARD'},
                {'ETag': '"cc1a0115345e38f58a88b83722a9d2d8"',
                 'Key': 'bert.zarr/.zmetadata',
                 'LastModified': '2019-06-24T20:43:40.862Z',
                 'Size': -1,
                 'StorageClass': 'STANDARD'},
            ],
            list_bucket_result.get('Contents')
        )
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
        list_bucket_result = self.list_bucket(delimiter='/',
                                              prefix='bert.zarr')
        self.assert_list_bucket_result(list_bucket_result,
                                       delimiter='/',
                                       prefix='bert.zarr')
        self.assertNotIn('Contents', list_bucket_result)
        self.assertIsInstance(list_bucket_result.get('CommonPrefixes'), list)
        self.assertEqual([{'Prefix': 'bert.zarr/'}],
                         list_bucket_result.get('CommonPrefixes'))


class ListBucketV1Test(ListS3BucketTest, ListS3BucketV12TestsMixin):
    def list_bucket(self, **kwargs):
        return list_s3_bucket_v1(self.object_storage,
                                 name='datasets',
                                 last_modified='2019-06-24T20:43:40.862Z',
                                 **kwargs)

    def test_list_bucket_v1_truncated(self):
        list_bucket_result = self.list_bucket(max_keys=5)
        self.assert_list_bucket_result(
            list_bucket_result,
            max_keys=5,
            is_truncated=True,
            next_marker='bert.zarr/lat/0'
        )
        self.assertIsInstance(list_bucket_result.get('Contents'), list)
        self.assertEqual(
            [
                {'ETag': '"16a49349c15b9cd60c5d8a05fc6ad649"',
                 'Key': 'bert.zarr/.zattrs',
                 'LastModified': '2019-06-24T20:43:40.862Z',
                 'Size': -1,
                 'StorageClass': 'STANDARD'},
                {'ETag': '"0a44cbe4de5e2936112efc2ed25b6223"',
                 'Key': 'bert.zarr/.zgroup',
                 'LastModified': '2019-06-24T20:43:40.862Z',
                 'Size': -1,
                 'StorageClass': 'STANDARD'},
                {'ETag': '"cc1a0115345e38f58a88b83722a9d2d8"',
                 'Key': 'bert.zarr/.zmetadata',
                 'LastModified': '2019-06-24T20:43:40.862Z',
                 'Size': -1,
                 'StorageClass': 'STANDARD'},
                {'ETag': '"b74fb939f20cf7cff02c208e8824faec"',
                 'Key': 'bert.zarr/lat/.zarray',
                 'LastModified': '2019-06-24T20:43:40.862Z',
                 'Size': -1,
                 'StorageClass': 'STANDARD'},
                {'ETag': '"05bbb0156ba6af07e613287e17c9385c"',
                 'Key': 'bert.zarr/lat/.zattrs',
                 'LastModified': '2019-06-24T20:43:40.862Z',
                 'Size': -1,
                 'StorageClass': 'STANDARD'}
            ],
            list_bucket_result.get('Contents')
        )
        self.assertNotIn('CommonPrefixes', list_bucket_result)

    def test_list_bucket_v1_result_to_xml(self):
        list_bucket_result = self.list_bucket(delimiter='/',
                                              max_keys=10,
                                              prefix='bibo.zarr/')
        self.assert_list_bucket_result(
            list_bucket_result,
            delimiter='/',
            max_keys=10,
            prefix='bibo.zarr/'
        )
        xml = list_bucket_result_to_xml(list_bucket_result)
        with open(os.path.join(os.path.dirname(__file__),
                               'res',
                               'list-bucket-v1-result.xml')) as fp:
            expected_xml = fp.read()
        self.assertEqual(expected_xml, xml)

    def assert_list_bucket_result(self,
                                  list_bucket_result,
                                  name="datasets",
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


class ListS3BucketV2Test(ListS3BucketTest, ListS3BucketV12TestsMixin):

    def test_list_bucket_v2_truncated(self):
        list_bucket_result = self.list_bucket(max_keys=5)
        self.assert_list_bucket_result(
            list_bucket_result,
            max_keys=5,
            is_truncated=True,
            next_continuation_token=6
        )
        self.assertIsInstance(list_bucket_result.get('Contents'), list)
        self.assertEqual(
            [
                {'ETag': '"16a49349c15b9cd60c5d8a05fc6ad649"',
                 'Key': 'bert.zarr/.zattrs',
                 'LastModified': '2019-06-24T20:43:40.862Z',
                 'Size': -1,
                 'StorageClass': 'STANDARD'},
                {'ETag': '"0a44cbe4de5e2936112efc2ed25b6223"',
                 'Key': 'bert.zarr/.zgroup',
                 'LastModified': '2019-06-24T20:43:40.862Z',
                 'Size': -1,
                 'StorageClass': 'STANDARD'},
                {'ETag': '"cc1a0115345e38f58a88b83722a9d2d8"',
                 'Key': 'bert.zarr/.zmetadata',
                 'LastModified': '2019-06-24T20:43:40.862Z',
                 'Size': -1,
                 'StorageClass': 'STANDARD'},
                {'ETag': '"b74fb939f20cf7cff02c208e8824faec"',
                 'Key': 'bert.zarr/lat/.zarray',
                 'LastModified': '2019-06-24T20:43:40.862Z',
                 'Size': -1,
                 'StorageClass': 'STANDARD'},
                {'ETag': '"05bbb0156ba6af07e613287e17c9385c"',
                 'Key': 'bert.zarr/lat/.zattrs',
                 'LastModified': '2019-06-24T20:43:40.862Z',
                 'Size': -1,
                 'StorageClass': 'STANDARD'}
            ],
            list_bucket_result.get('Contents')
        )
        self.assertNotIn('CommonPrefixes', list_bucket_result)

    def test_list_bucket_v2_result_to_xml(self):
        list_bucket_result = self.list_bucket(delimiter='/',
                                              max_keys=10,
                                              prefix='bibo.zarr/')
        self.assert_list_bucket_result(
            list_bucket_result,
            delimiter='/',
            max_keys=10,
            prefix='bibo.zarr/'
        )
        xml = list_bucket_result_to_xml(list_bucket_result)
        with open(os.path.join(os.path.dirname(__file__),
                               'res',
                               'list-bucket-v2-result.xml')) as fp:
            expected_xml = fp.read()
        self.assertEqual(expected_xml, xml)

    def list_bucket(self, **kwargs):
        return list_s3_bucket_v2(self.object_storage,
                                 name='datasets',
                                 last_modified='2019-06-24T20:43:40.862Z',
                                 **kwargs)

    def assert_list_bucket_result(self,
                                  list_bucket_result,
                                  name="datasets",
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
        self.assertEqual(continuation_token,
                         list_bucket_result.get('ContinuationToken'))
        self.assertEqual(next_continuation_token,
                         list_bucket_result.get('NextContinuationToken'))


EXPECTED_KEYS = [
    'bert.zarr/',
    'bert.zarr/.zattrs',
    'bert.zarr/.zgroup',
    'bert.zarr/.zmetadata',
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
    'bibo.zarr/.zmetadata',
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
