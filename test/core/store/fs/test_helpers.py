import pathlib
import unittest

import fsspec
import fsspec.implementations.local

from xcube.core.store.fs.helpers import get_fs_path_class
from xcube.core.store.fs.helpers import resolve_path


class FsHelpersTest(unittest.TestCase):

    def test_resolve_path_succeeds(self):
        path = pathlib.PurePosixPath("a", "b", "c")
        self.assertEqual(pathlib.PurePosixPath("a", "b", "c"),
                         resolve_path(path))

        path = pathlib.PurePosixPath(".", "b", "c")
        self.assertEqual(pathlib.PurePosixPath("b", "c"),
                         resolve_path(path))

        path = pathlib.PurePosixPath("a", ".", "c")
        self.assertEqual(pathlib.PurePosixPath("a", "c"),
                         resolve_path(path))

        path = pathlib.PurePosixPath("a", "b", ".")
        self.assertEqual(pathlib.PurePosixPath("a", "b"),
                         resolve_path(path))

        path = pathlib.PurePosixPath("a", "..", "c")
        self.assertEqual(pathlib.PurePosixPath("c"),
                         resolve_path(path))

        path = pathlib.PurePosixPath("a", "b", "..")
        self.assertEqual(pathlib.PurePosixPath("a"),
                         resolve_path(path))

    def test_resolve_path_fails(self):
        path = pathlib.PurePosixPath("..", "b", "c")
        with self.assertRaises(ValueError) as cm:
            resolve_path(path)
        self.assertEqual('cannot resolve path, misplaced ".."',
                         f'{cm.exception}')

        path = pathlib.PurePosixPath("a", "..", "..")
        with self.assertRaises(ValueError) as cm:
            resolve_path(path)
        self.assertEqual('cannot resolve path, misplaced ".."',
                         f'{cm.exception}')

    def test_get_fs_path_class(self):
        self.assertIs(pathlib.PurePath,
                      get_fs_path_class(
                          fsspec.implementations.local.LocalFileSystem()
                      ))
        self.assertIs(pathlib.PurePath,
                      get_fs_path_class(fsspec.filesystem("file")))
        self.assertIs(pathlib.PurePosixPath,
                      get_fs_path_class(fsspec.filesystem("s3")))
        self.assertIs(pathlib.PurePosixPath,
                      get_fs_path_class(fsspec.filesystem("memory")))
