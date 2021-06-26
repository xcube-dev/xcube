import os
import os.path
import os.path
import unittest

from xcube.core.byoa.constants import DEFAULT_TEMP_FILE_PREFIX
from xcube.core.byoa import FileSet

PARENT_DIR = os.path.dirname(__file__)


class FileSetTest(unittest.TestCase):

    def test_is_local(self):
        self.assertTrue(FileSet('test_data/user_code').is_local())
        self.assertTrue(FileSet('file://test_data/user_code').is_local())
        self.assertFalse(FileSet('s3://xcube/user_code').is_local())
        self.assertFalse(FileSet('https://xcube/user_code.zip').is_local())
        self.assertFalse(FileSet('github://dcs4cop:xcube@v0.8.1').is_local())

    def test_is_local_dir(self):
        local_dir_path = os.path.join(PARENT_DIR, 'test_data', 'user_code')
        self.assertTrue(FileSet(local_dir_path).is_local_dir())
        self.assertTrue(FileSet('file://' + local_dir_path).is_local_dir())
        self.assertFalse(FileSet('s3://eurodatacube/test/').is_local_dir())
        self.assertFalse(FileSet('s3://eurodatacube/test.zip').is_local_dir())

    def test_is_local_zip(self):
        local_zip_path = os.path.join(PARENT_DIR, 'test_data', 'user_code.zip')
        self.assertTrue(FileSet(local_zip_path).is_local_zip())
        self.assertTrue(FileSet('file://' + local_zip_path).is_local_zip())
        self.assertFalse(FileSet('s3://eurodatacube/test/').is_local_zip())
        self.assertFalse(FileSet('s3://eurodatacube/test.zip').is_local_zip())

    def test_keys(self):
        self._test_keys_for_local('test_data/user_code')
        self._test_keys_for_local('test_data/user_code.zip')

    def _test_keys_for_local(self, rel_path: str):
        path = os.path.join(PARENT_DIR, rel_path)

        file_set = FileSet(path)
        self.assertEqual(
            {
                'NOTES.md',
                'processor.py',
                'impl/__init__.py',
                'impl/algorithm.py',
            },
            set(file_set.keys()))

        file_set = FileSet(path, includes=['*.py'])
        self.assertEqual(
            {
                'processor.py',
                'impl/__init__.py',
                'impl/algorithm.py',
            },
            set(file_set.keys()))

        file_set = FileSet(path, excludes=['NOTES.md'])
        self.assertEqual(
            {
                'processor.py',
                'impl/__init__.py',
                'impl/algorithm.py',
            },
            set(file_set.keys()))

    def test_to_local_zip_then_to_local_dir(self):
        dir_path = os.path.join(PARENT_DIR, 'test_data', 'user_code')

        zip_file_set = FileSet(dir_path).to_local_zip()
        self.assertIsInstance(zip_file_set, FileSet)
        self.assertTrue(zip_file_set.is_local())
        self.assertFalse(zip_file_set.is_local_dir())
        self.assertTrue(zip_file_set.is_local_zip())
        self.assertRegex(zip_file_set.path,
                         f'^.*{DEFAULT_TEMP_FILE_PREFIX}.*\\.zip$')
        self.assertEqual(
            {
                'NOTES.md',
                'processor.py',
                'impl/__init__.py',
                'impl/algorithm.py',
            },
            set(zip_file_set.keys())
        )

        dir_file_set = zip_file_set.to_local_dir()
        self.assertIsInstance(dir_file_set, FileSet)
        self.assertTrue(zip_file_set.is_local())
        self.assertTrue(dir_file_set.is_local_dir())
        self.assertFalse(dir_file_set.is_local_zip())
        self.assertRegex(dir_file_set.path,
                         f'^.*{DEFAULT_TEMP_FILE_PREFIX}.*$')
        self.assertEqual(
            {
                'NOTES.md',
                'processor.py',
                'impl/__init__.py',
                'impl/algorithm.py',
            },
            set(dir_file_set.keys())
        )

    def test_to_local_dir_then_to_local_zip(self):
        zip_path = os.path.join(PARENT_DIR, 'test_data', 'user_code.zip')

        dir_file_set = FileSet(zip_path).to_local_dir()
        self.assertIsInstance(dir_file_set, FileSet)
        self.assertTrue(dir_file_set.is_local())
        self.assertTrue(dir_file_set.is_local_dir())
        self.assertFalse(dir_file_set.is_local_zip())
        self.assertRegex(dir_file_set.path,
                         f'^.*{DEFAULT_TEMP_FILE_PREFIX}.*$')
        self.assertEqual(
            {
                'NOTES.md',
                'processor.py',
                'impl/__init__.py',
                'impl/algorithm.py',
            },
            set(dir_file_set.keys())
        )

        zip_file_set = dir_file_set.to_local_zip()
        self.assertIsInstance(zip_file_set, FileSet)
        self.assertTrue(dir_file_set.is_local())
        self.assertFalse(zip_file_set.is_local_dir())
        self.assertTrue(zip_file_set.is_local_zip())
        self.assertRegex(zip_file_set.path,
                         f'^.*{DEFAULT_TEMP_FILE_PREFIX}.*\\.zip$')
        self.assertEqual(
            {
                'NOTES.md',
                'processor.py',
                'impl/__init__.py',
                'impl/algorithm.py',
            },
            set(zip_file_set.keys())
        )

    def test_to_dict(self):
        file_set = FileSet('test_data/user_code')
        self.assertEqual(
            {'path': 'test_data/user_code'},
            file_set.to_dict())

        file_set = FileSet('test_data/user_code',
                           includes=['*.py'])
        self.assertEqual(
            {
                'path': 'test_data/user_code',
                'includes': ['*.py']
            },
            file_set.to_dict())

        file_set = FileSet('test_data/user_code',
                           excludes=['NOTES.md'])
        self.assertEqual(
            {
                'path': 'test_data/user_code',
                'excludes': ['NOTES.md']
            },
            file_set.to_dict())

        file_set = FileSet('s3://xcube/user_code',
                           storage_params={'anon': True},
                           includes=['*.py'],
                           excludes=['NOTES.md'])
        self.assertEqual(
            {
                'path': 's3://xcube/user_code',
                'storage_params': {'anon': True},
                'includes': ['*.py'],
                'excludes': ['NOTES.md'],
            },
            file_set.to_dict())
