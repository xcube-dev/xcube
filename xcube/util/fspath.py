# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import pathlib
from typing import Type
from collections.abc import Iterator

import fsspec
from fsspec.implementations.local import LocalFileSystem


def is_local_fs(fs: fsspec.AbstractFileSystem) -> bool:
    """Check whether *fs* is a local filesystem."""
    return "file" in fs.protocol or isinstance(fs, LocalFileSystem)


def get_fs_path_class(fs: fsspec.AbstractFileSystem) -> type[pathlib.PurePath]:
    """Get the appropriate ``pathlib.PurePath`` class for the filesystem *fs*."""
    if is_local_fs(fs):
        # Will return PurePosixPath or a PureWindowsPath object
        return pathlib.PurePath
    # PurePosixPath for non-local filesystems such as S3,
    # so we force use of forward slashes as separators
    return pathlib.PurePosixPath


def resolve_path(path: pathlib.PurePath) -> pathlib.PurePath:
    """Resolve "." and ".." occurrences in *path* without I/O and
    return a new path.
    """
    reversed_parts = reversed(path.parts)
    reversed_norm_parts = list(_resolve_path_impl(reversed_parts))
    parts = reversed(reversed_norm_parts)
    return type(path)(*parts)


def _resolve_path_impl(reversed_parts: Iterator[str]):
    skips = False
    for part in reversed_parts:
        if part == ".":
            continue
        elif part == "..":
            skips += 1
            continue
        if skips == 0:
            yield part
        elif skips > 0:
            skips -= 1
    if skips != 0:
        raise ValueError('cannot resolve path, misplaced ".."')
