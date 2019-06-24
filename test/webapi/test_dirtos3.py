import json
import unittest

from xcube.api import new_cube
from xcube.api.readwrite import write_cube
from xcube.util.dsio import rimraf


class DirToS3Test(unittest.TestCase):

    def test_it(self):
        self.maxDiff = None
        result = dir_to_s3(self.TEST_CUBE, self.TEST_CUBE, '')
        print(json.dumps(result, indent=2))
        self.assertIsInstance(result, dict)
        self.assertIn('ListBucketResult', result)
        self.assertIsInstance(result['ListBucketResult'], dict)
        self.assertIn('Contents', result['ListBucketResult'])
        self.assertIsInstance(result['ListBucketResult']['Contents'], list)
        self.assertEqual(85, len(result['ListBucketResult']['Contents']))

    TEST_CUBE = "test.zarr"

    @classmethod
    def setUpClass(cls) -> None:
        rimraf(cls.TEST_CUBE)
        cube = new_cube(time_periods=3,
                        variables=dict(precipitation=0.9,
                                       temperature=278.3)).chunk(dict(time=1, lat=90, lon=90))
        write_cube(cube, cls.TEST_CUBE, "zarr", cube_asserted=True)

    @classmethod
    def tearDownClass(cls) -> None:
        rimraf(cls.TEST_CUBE)


import os.path
import datetime


def dir_to_s3(bucket_name, dir_path, rel_path, delimiter='/', prefix=None, max_keys=1000, marker=None):
    full_path = os.path.abspath(os.path.normpath(os.path.join(dir_path, rel_path)))
    if os.path.isfile(full_path):
        # Get object
        return None
    else:
        contents_list = []
        is_truncated = False
        next_marker = None
        marker_seen = marker is None

        for root, dirs, files in os.walk(full_path):
            common_prefix = os.path.commonprefix([full_path, root])
            key_prefix = root[len(common_prefix) + 1:].replace('\\', '/') + '/'

            keys_and_paths = [(key_prefix, root)]
            for file in files:
                keys_and_paths.append((key_prefix + file, os.path.join(root, file)))

            for key, path in keys_and_paths:

                if key == marker:
                    marker_seen = True

                if not marker_seen:
                    continue

                if len(contents_list) == max_keys:
                    is_truncated = True
                    next_marker = key
                    break

                stat = os.stat(os.path.join(root))
                contents_list.append(dict(Key=key,
                                          Size=stat.st_size,
                                          LastModified=str(datetime.datetime.fromtimestamp(stat.st_mtime)),
                                          ETag=hex(abs(hash(root + key)))[2:],
                                          StorageClass='STANDARD'))

        return dict(
            ListBucketResult=dict(
                Name=bucket_name,
                Prefix=prefix,
                Delimiter=delimiter,
                MaxKeys=max_keys,
                IsTruncated=is_truncated,
                Marker=marker,
                NextMarker=next_marker,
                Contents=contents_list
            )
        )
