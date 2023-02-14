import pathlib
import unittest

import fsspec
import fsspec.implementations.local

from xcube.util.fspath import get_fs_path_class
from xcube.util.fspath import resolve_path


class FsPathTest(unittest.TestCase):
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

    def test_resolve_path_succeeds(self):
        path = pathlib.PurePosixPath("a", "b", "c")
        self.assertEqual(pathlib.PurePosixPath("a", "b", "c"),
                         resolve_path(path))

        path = pathlib.PurePosixPath("../core/store/fs", "b", "c")
        self.assertEqual(pathlib.PurePosixPath("b", "c"),
                         resolve_path(path))

        path = pathlib.PurePosixPath("a", "../core/store/fs", "c")
        self.assertEqual(pathlib.PurePosixPath("a", "c"),
                         resolve_path(path))

        path = pathlib.PurePosixPath("a", "b", "../core/store/fs")
        self.assertEqual(pathlib.PurePosixPath("a", "b"),
                         resolve_path(path))

        path = pathlib.PurePosixPath("a", "../core/store", "c")
        self.assertEqual(pathlib.PurePosixPath("c"),
                         resolve_path(path))

        path = pathlib.PurePosixPath("a", "b", "../core/store")
        self.assertEqual(pathlib.PurePosixPath("a"),
                         resolve_path(path))

    def test_resolve_path_fails(self):
        path = pathlib.PurePosixPath("../core/store", "b", "c")
        with self.assertRaises(ValueError) as cm:
            resolve_path(path)
        self.assertEqual('cannot resolve path, misplaced ".."',
                         f'{cm.exception}')

        path = pathlib.PurePosixPath("a", "../core/store", "..")
        with self.assertRaises(ValueError) as cm:
            resolve_path(path)
        self.assertEqual('cannot resolve path, misplaced ".."',
                         f'{cm.exception}')

