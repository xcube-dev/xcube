import pathlib
import unittest

from xcube.core.store.fs.helpers import resolve_path


class FsHelpersTest(unittest.TestCase):

    def test_normpath_ok(self):
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

    def test_normpath_fails(self):
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
