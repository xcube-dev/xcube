import atexit
import os
import os.path
import os.path
import shutil
import unittest

from xcube.core.gen2.byoa.fileset import FileSet


class FileSetTest(unittest.TestCase):
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
                           parameters={'anon': True},
                           includes=['*.py'],
                           excludes=['NOTES.md'])
        self.assertEqual(
            {
                'path': 's3://xcube/user_code',
                'parameters': {'anon': True},
                'includes': ['*.py'],
                'excludes': ['NOTES.md'],
            },
            file_set.to_dict())

    def test_keys(self):
        self._test_keys_for_local_dir('test_data/user_code')
        self._test_keys_for_local_dir('test_data/user_code.zip')

    def _test_keys_for_local_dir(self, rel_path: str):

        base_dir = os.path.join(os.path.dirname(__file__), rel_path)

        file_set = FileSet(base_dir)
        self.assertEqual(
            {
                'NOTES.md',
                'processor.py',
                'impl/__init__.py',
                'impl/algorithm.py',
            },
            set(file_set.keys()))

        file_set = FileSet(base_dir, includes=['*.py'])
        self.assertEqual(
            {
                'processor.py',
                'impl/__init__.py',
                'impl/algorithm.py',
            },
            set(file_set.keys()))

        file_set = FileSet(base_dir, excludes=['NOTES.md'])
        self.assertEqual(
            {
                'processor.py',
                'impl/__init__.py',
                'impl/algorithm.py',
            },
            set(file_set.keys()))

    def test_zip_and_unzip(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'test_data/user_code')
        zip_file_set = FileSet(base_dir).to_local_zip()
        try:
            self.assertIsInstance(zip_file_set, FileSet)
            self.assertTrue(zip_file_set.path.endswith('.zip'))
            self.assertFalse(zip_file_set.is_local_dir())
            self.assertTrue(zip_file_set.is_local_zip())
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
            try:
                self.assertIsInstance(dir_file_set, FileSet)
                self.assertFalse(dir_file_set.path.endswith('.zip'))
                self.assertTrue(dir_file_set.is_local_dir())
                self.assertFalse(dir_file_set.is_local_zip())
                self.assertEqual(
                    {
                        'NOTES.md',
                        'processor.py',
                        'impl/__init__.py',
                        'impl/algorithm.py',
                    },
                    set(dir_file_set.keys())
                )
            finally:
                def del_dir(dir_path):
                    shutil.rmtree(dir_path)

                atexit.register(del_dir, dir_file_set.path)
        finally:
            def del_file(file_path):
                os.remove(file_path)

            atexit.register(del_file, zip_file_set.path)
