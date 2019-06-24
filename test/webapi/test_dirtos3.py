import unittest
import json

from xcube.api import new_cube
from xcube.api.readwrite import write_cube
from xcube.util.dsio import rimraf


class DirToS3Test(unittest.TestCase):

    def test_it(self):
        self.maxDiff = None
        result = dir_to_s3(self.TEST_CUBE, '')
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

def dir_to_s3(dir_path, rel_path, delimiter='/', prefix=None, max_keys=1000, marker=None):
    full_path = os.path.abspath(os.path.normpath(os.path.join(dir_path, rel_path)))
    if os.path.isfile(full_path):
        # Get object
        return None
    else:
        i = 0
        contents_list = []
        for root, dirs, files in os.walk(full_path):
            common_prefix = os.path.commonprefix([full_path, root])
            key_prefix = root[len(common_prefix) + 1:].replace('\\', '/') + '/'

            key = key_prefix
            stat = os.stat(os.path.join(root))
            contents_list.append(dict(Key=key,
                                      Size=stat.st_size,
                                      LastModified=str(datetime.datetime.fromtimestamp(stat.st_mtime)),
                                      ETag=hex(abs(hash(root + key)))[2:],
                                      StorageClass='STANDARD'))
            if i >= max_keys:
                break
            i += 1
            for file in files:
                stat = os.stat(os.path.join(root, file))
                key = key_prefix + file
                contents_list.append(dict(Key=key,
                                          Size=stat.st_size,
                                          LastModified=str(datetime.datetime.fromtimestamp(stat.st_mtime)),
                                          ETag=hex(abs(hash(root + key)))[2:],
                                          StorageClass='STANDARD'))
                if i >= max_keys:
                    break
                i += 1

        return dict(ListBucketResult=dict(Contents=contents_list))