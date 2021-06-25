import fsspec
import unittest
import zipfile
import os.path

PARENT_DIR = os.path.dirname(__file__)

CACHE_PATH = os.path.join(PARENT_DIR, '_fsspec_cache')
CACHE_PARAMS = dict(
    cache_storage=CACHE_PATH,
)


@unittest.skipIf(True, 'For testing fsspec behaviour only.')
class FileSystemSpecTest(unittest.TestCase):
    def test_https_zip_in_cache(self):
        url = 'simplecache::https://github.com/dcs4cop/xcube/archive/v0.8.1.zip'
        with fsspec.open(url, simplecache=CACHE_PARAMS) as f:
            print(f.name)
            self.assertTrue(f.name.startswith(CACHE_PATH))
            self.assertTrue(zipfile.is_zipfile(f))
            with zipfile.ZipFile(f, 'r') as zf:
                files = zf.filelist
                self.assertTrue(len(files) > 0)
        path = fsspec.open_local(url, simplecache=CACHE_PARAMS)
        print(path)
        self.assertTrue(path.startswith(CACHE_PATH))
        self.assertTrue(zipfile.is_zipfile(path))
        with zipfile.ZipFile(path, 'r') as zf:
            files = zf.filelist
            self.assertTrue(len(files) > 0)

    def test_s3_in_cache(self):
        url = 'simplecache::s3://xcube-examples/OLCI-SNS-RAW-CUBE-2.zarr'
        with fsspec.open(url, simplecache=CACHE_PARAMS) as f:
            print(f.name)
            self.assertTrue(f.name.startswith(CACHE_PATH))
            self.assertTrue(os.path.isdir(f))
        path = fsspec.open_local(url, simplecache=CACHE_PARAMS)
        print(path)
        self.assertTrue(path.startswith(CACHE_PATH))
        self.assertTrue(os.path.isdir(path))
