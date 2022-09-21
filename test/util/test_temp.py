import os.path
import unittest

from xcube.util.temp import new_temp_dir
from xcube.util.temp import new_temp_file
from xcube.util.temp import remove_dir_later
from xcube.util.temp import remove_file_later


class TempTest(unittest.TestCase):
    def test_new_temp_dir(self):
        dir_path = new_temp_dir()
        self.assertIsInstance(dir_path, str)
        self.assertTrue(os.path.basename(dir_path).startswith('xcube-'))
        self.assertTrue(os.path.isabs(dir_path))
        self.assertTrue(os.path.isdir(dir_path))

        dir_path = new_temp_dir(prefix='bibo.', suffix='.zarr')
        self.assertIsInstance(dir_path, str)
        self.assertTrue(os.path.basename(dir_path).startswith('bibo.'))
        self.assertTrue(os.path.basename(dir_path).endswith('.zarr'))
        self.assertTrue(os.path.isabs(dir_path))
        self.assertTrue(os.path.isdir(dir_path))

    def test_new_temp_file(self):
        fd, file_path = new_temp_file()
        self.assertIsInstance(fd, int)
        self.assertIsInstance(file_path, str)
        self.assertTrue(os.path.basename(file_path).startswith('xcube-'))
        self.assertTrue(os.path.isabs(file_path))
        self.assertTrue(os.path.isfile(file_path))

        fd, file_path = new_temp_file(prefix='bibo.', suffix='.zip')
        self.assertIsInstance(fd, int)
        self.assertIsInstance(file_path, str)
        self.assertTrue(os.path.basename(file_path).startswith('bibo.'))
        self.assertTrue(os.path.basename(file_path).endswith('.zip'))
        self.assertTrue(os.path.isabs(file_path))
        self.assertTrue(os.path.isfile(file_path))

    def test_remove_file_later(self):
        with open('__test__.txt', 'w') as stream:
            stream.write('Hello!')
        file_path = remove_file_later('__test__.txt')
        self.assertTrue(os.path.isabs(file_path))

    def test_remove_dir_later(self):
        if not os.path.isdir('__test__'):
            os.mkdir('__test__')
        dir_path = remove_dir_later('__test__')
        self.assertTrue(os.path.isabs(dir_path))
