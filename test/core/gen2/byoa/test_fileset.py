import os.path
import os.path
import unittest
import zipfile

from xcube.core.gen2.byoa.fileset import FileSet


class FileSetTest(unittest.TestCase):

    def test_files(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'user_code')

        self.assertEqual(
            {
                'NOTES.md',
                '__init__.py',
                'processor.py',
                'impl/__init__.py',
                'impl/algorithm.py',
            },
            set(os.path.relpath(f, base_dir).replace(os.path.sep, '/')
                for f in FileSet(base_dir).files))

        self.assertEqual(
            {
                '__init__.py',
                'processor.py',
                'impl/__init__.py',
                'impl/algorithm.py',
            },
            set(os.path.relpath(f, base_dir).replace(os.path.sep, '/')
                for f in FileSet(base_dir, includes=['*.py']).files))

        self.assertEqual(
            {
                '__init__.py',
                'processor.py',
                'impl/__init__.py',
                'impl/algorithm.py',
            },
            set(os.path.relpath(f, base_dir).replace(os.path.sep, '/')
                for f in FileSet(base_dir, excludes=['NOTES.md']).files))

    def test_to_dict(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'user_code')

        self.assertEqual({'base_dir': base_dir},
                         FileSet(base_dir).to_dict())

        self.assertEqual({'base_dir': base_dir,
                          'includes': ['*.py']},
                         FileSet(base_dir,
                                 includes=['*.py']).to_dict())

        self.assertEqual({'base_dir': base_dir,
                          'excludes': ['NOTES.md']},
                         FileSet(base_dir,
                                 excludes=['NOTES.md']).to_dict())

    def test_zip(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'user_code')
        zip_path = FileSet(base_dir).zip()
        self.assertIsInstance(zip_path, str)
        self.assertTrue(os.path.isfile(zip_path))
        self.assertTrue(zip_path.endswith('.zip'))
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                self.assertEqual(
                    {
                        'NOTES.md',
                        '__init__.py',
                        'processor.py',
                        'impl/__init__.py',
                        'impl/algorithm.py',
                    },
                    set(f.filename for f in zip_file.filelist)
                )
        finally:
            os.remove(zip_path)
