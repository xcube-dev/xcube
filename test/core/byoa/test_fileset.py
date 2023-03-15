import os
import os.path
import unittest
import zipfile

import fsspec

from xcube.core.byoa import FileSet
from xcube.core.byoa.constants import TEMP_FILE_PREFIX
from xcube.core.dsio import rimraf

PARENT_DIR = os.path.dirname(__file__)


class FileSetToLocalTest(unittest.TestCase):
    test_dir = os.path.join(os.path.dirname(__file__), 'test-data')

    def setUp(self) -> None:
        if os.path.exists(self.test_dir):
            rimraf(self.test_dir)
        os.mkdir(self.test_dir)

    def tearDown(self) -> None:
        rimraf(self.test_dir)

    def _make_dir(self):
        for i in range(3):
            file_path = f'{self.test_dir}/module_{i}.py'
            with open(file_path, 'w') as file:
                file.write('\n')
        return self.test_dir

    def _make_zip(self, prefix: str = ''):
        zip_path = f'{self.test_dir}/modules.zip'
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for i in range(3):
                file_path = f'{(prefix + "/") if prefix else ""}module_{i}.py'
                with zf.open(file_path, 'w') as file:
                    file.write(b'\n')
        return zip_path

    def test_to_local_from_local_dir(self):
        dir_path = self._make_dir()
        file_set = FileSet(dir_path)
        local_file_set = file_set.to_local()
        self.assertIs(local_file_set, file_set)

    def test_to_local_from_memory_dir(self):
        mem_fs: fsspec.AbstractFileSystem = fsspec.filesystem("memory")
        mem_fs.mkdir("package")
        mem_fs.touch("package/__init__.py")
        mem_fs.touch("package/module1.py")
        mem_fs.mkdir("package/module2")
        mem_fs.touch("package/module2/__init__.py")
        file_set = FileSet("memory://package")
        local_file_set = file_set.to_local()
        self.assertTrue(os.path.isdir(local_file_set.path))
        self.assertTrue(os.path.isfile(local_file_set.path
                                       + "/__init__.py"))
        self.assertTrue(os.path.isfile(local_file_set.path
                                       + "/module1.py"))
        self.assertTrue(os.path.isfile(local_file_set.path
                                       + "/module2/__init__.py"))

    def test_to_local_from_local_flat_zip(self):
        zip_path = self._make_zip()
        file_set = FileSet(zip_path)
        local_file_set = file_set.to_local()
        self.assertIs(local_file_set, file_set)

    def test_to_local_from_local_nested_zip(self):
        zip_path = self._make_zip(prefix='content')
        file_set = FileSet(zip_path, sub_path='content')
        local_file_set = file_set.to_local()
        self.assertIs(local_file_set, file_set)
        # self.assertNotEqual(local_file_set.path, file_set.path)
        # self.assertTrue(os.path.isdir(local_file_set.path))
        # for i in range(3):
        #     file_path = f'{local_file_set.path}/module_{i}.py'
        #     self.assertTrue(os.path.isfile(file_path))


class FileSetTest(unittest.TestCase):

    def test_is_local(self):
        self.assertTrue(FileSet('test_data/user_code').is_local())
        self.assertTrue(FileSet('test_data/user_code.zip').is_local())
        self.assertTrue(FileSet('file://test_data/user_code').is_local())
        self.assertTrue(FileSet('file://test_data/user_code.zip').is_local())
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

        file_set = FileSet(path, excludes=['__pycache__'])
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

        file_set = FileSet(path, excludes=['NOTES.md', '__pycache__'])
        self.assertEqual(
            {
                'processor.py',
                'impl/__init__.py',
                'impl/algorithm.py',
            },
            set(file_set.keys()))

    def test_to_local_zip_then_to_local_dir(self):
        dir_path = os.path.join(PARENT_DIR, 'test_data', 'user_code')

        file_set = FileSet(dir_path, excludes=['__pycache__'])
        zip_file_set = file_set.to_local_zip()
        self.assertIsInstance(zip_file_set, FileSet)
        self.assertTrue(zip_file_set.is_local())
        self.assertFalse(zip_file_set.is_local_dir())
        self.assertTrue(zip_file_set.is_local_zip())
        self.assertRegex(zip_file_set.path,
                         f'^.*{TEMP_FILE_PREFIX}.*\\.zip$')
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
                         f'^.*{TEMP_FILE_PREFIX}.*$')
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
                         f'^.*{TEMP_FILE_PREFIX}.*$')
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
                         f'^.*{TEMP_FILE_PREFIX}.*\\.zip$')
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
                           sub_path='test',
                           storage_params={'anon': True},
                           includes=['*.py'],
                           excludes=['NOTES.md'])
        self.assertEqual(
            {
                'path': 's3://xcube/user_code',
                'sub_path': 'test',
                'storage_params': {'anon': True},
                'includes': ['*.py'],
                'excludes': ['NOTES.md'],
            },
            file_set.to_dict())
