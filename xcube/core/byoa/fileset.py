# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import fnmatch
import os.path
import re
import zipfile
from typing import Any, Dict, Optional, List, Union
from collections.abc import Collection, Iterator

import fsspec
import fsspec.implementations.zip
from fsspec.implementations.local import LocalFileSystem

from xcube.core.byoa.constants import TEMP_FILE_PREFIX
from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonObject
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema
from xcube.util.temp import new_temp_dir
from xcube.util.temp import new_temp_file


class _FileSetDetails:
    def __init__(
        self, fs: fsspec.AbstractFileSystem, root: str, local_path: Union[None, str]
    ):
        """Stores file system details about a FileSet instance.
        Internal helper class.

        Args:
            fs: file system
            root: root in file system
            local_path: the local or network path. None, if this is not
                a local path.
        """
        self._fs = fs
        self._root = root
        self._local_path = local_path
        self._mapper: Optional[fsspec.FSMap] = None

    @classmethod
    def new(cls, path: str, storage_params: dict[str, Any] = None) -> "_FileSetDetails":
        try:
            fs, root = fsspec.core.url_to_fs(path, **(storage_params or {}))
        except (ImportError, OSError) as e:
            raise ValueError(f"Illegal file set {path!r}") from e
        local_path = (
            root if "file" in fs.protocol or isinstance(fs, LocalFileSystem) else None
        )
        return _FileSetDetails(fs, root, local_path)

    @property
    def fs(self) -> fsspec.AbstractFileSystem:
        return self._fs

    @property
    def root(self) -> str:
        return self._root

    @property
    def local_path(self) -> Optional[str]:
        return self._local_path

    @property
    def mapper(self) -> fsspec.FSMap:
        if self._mapper is None:
            fs, root = self.fs, self.root
            if self.local_path and zipfile.is_zipfile(self.local_path):
                fs = fsspec.implementations.zip.ZipFileSystem(self.local_path)
                root = ""
            self._mapper = fsspec.FSMap(root, fs)
        return self._mapper


class FileSet(JsonObject):
    """A set of files that can found at some abstract root *path*.
    The *path* may identify local or remote filesystems.
    A filesystem may require specific *parameters*, e.g.
    user credentials.

    This implementation is based on the ``fsspec``
    package and the documentation about the different
    possible file systems and their specific parameters
    can be found at https://filesystem-spec.readthedocs.io/.

    Examples:

        FileSet('eurodatacube/my_processor',
                includes=['*.py'])

        FileSet('eurodatacube/my_processor.zip',
                excludes=['*.zarr'])

        FileSet('s3://eurodatacube/my_processor.zip',
                storage_params=dict(key='...',
                                    secret='...')

        FileSet('github://dcs4cop:xcube@v0.8.1/',
                storage_params=dict(username='...',
                                    token='ghp_...')

    If *includes* wildcard patterns are given, only files
    whose paths match any of the patterns are included.
    If *excludes* wildcard patterns are given, any file
    matching one of the patterns is excluded.
    The *includes*, if any, are applied before the
    *excludes*, if any.

    Args:
        path: Root path of the file set.
        sub_path: An optional sub-path relative to *path*. If given,
            this is where the actual file sets starts. For example, the
            only top-level directory entry in a ZIP archive. In archive
            files from GitHub this would be typically "<gh_repo>-<gh-
            release-name>". Note: wildcard characters are not yet
            allowed.
        includes: Wildcard patterns used to include a file.
        excludes: Wildcard patterns used to exclude a file.
        storage_params: File system specific storage parameters.
    """

    def __init__(
        self,
        path: str,
        sub_path: str = None,
        includes: Collection[str] = None,
        excludes: Collection[str] = None,
        storage_params: dict[str, Any] = None,
    ):
        assert_instance(path, str, "path")
        assert_given(path, "path")
        if sub_path is not None:
            assert_instance(sub_path, str, "sub_path")
        self._path = path
        self._sub_path = sub_path
        self._storage_params = (
            dict(storage_params) if storage_params is not None else None
        )
        self._includes = list(includes) if includes is not None else None
        self._excludes = list(excludes) if excludes is not None else None
        # computed members
        self._include_patterns = _translate_patterns(includes or [])
        self._exclude_patterns = _translate_patterns(excludes or [])
        # cached, computed members
        self._details: Optional[_FileSetDetails] = None

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        """Get the JSON-schema for FileSet objects."""
        return JsonObjectSchema(
            properties=dict(
                path=JsonStringSchema(min_length=1),
                sub_path=JsonStringSchema(min_length=1),
                storage_params=JsonObjectSchema(additional_properties=True),
                includes=JsonArraySchema(items=JsonStringSchema(min_length=1)),
                excludes=JsonArraySchema(items=JsonStringSchema(min_length=1)),
            ),
            additional_properties=False,
            required=["path"],
            factory=cls,
        )

    @property
    def path(self) -> str:
        """Get the root path."""
        return self._path

    @property
    def sub_path(self) -> Optional[str]:
        """Get the sub-path relative to *path*."""
        return self._sub_path

    @property
    def storage_params(self) -> Optional[dict[str, Any]]:
        """Get optional parameters for the file system
        :attribute:path is referring to.
        """
        return self._storage_params

    @property
    def includes(self) -> Optional[list[str]]:
        """Wildcard patterns used to include a file."""
        return self._includes

    @property
    def excludes(self) -> Optional[list[str]]:
        """Wildcard patterns used to exclude a file."""
        return self._excludes

    def keys(self) -> Iterator[str]:
        """Get keys in this file set.
        A key is a normalized path relative to the root *path*.
        The forward slash "/" is used as path separator.
        """
        for key in self._get_details().mapper.keys():
            # print('-->', key)
            if self._accepts_key(key):
                yield key

    def get_local_path(self) -> Optional[str]:
        """Get the local path, if this file set is local,
        otherwise return None.
        """
        return self._get_details().local_path

    def is_local(self) -> bool:
        """Test whether this file set refers to a local file or directory.
        The test is made on this path's protocol.
        """
        if self._details is not None:
            return self._details.local_path is not None
        protocol, _ = fsspec.core.split_protocol(self._path)
        return protocol is None or protocol == "file"

    def to_local(self) -> "FileSet":
        """Turn this file set into a locally existing file set."""
        if self.is_local():
            return self

        details = self._get_details()
        fs, root = details.fs, details.root

        url_path = fsspec.core.strip_protocol(self.path)
        temp_file_suffix = ""
        for temp_file_suffix in reversed(url_path.split("/")):
            if temp_file_suffix != "":
                break

        if fs.isdir(root):
            temp_dir = new_temp_dir(prefix=TEMP_FILE_PREFIX, suffix=temp_file_suffix)

            # Note, actually want to use
            #    fs.get(root + "/*", temp_dir, recursive=True)
            # here, but fs.get() behaves unpredictably with
            # fsspec=2023.3.0 and s3fs=2023.3.0.
            # Sometimes it will create a new subdirectory in temp_dir
            # named after last path element of root, sometimes not.
            # Behaviour randomly changes if
            #   - temp_dir exists or not
            #   - root or temp_dir paths end with a slash
            #   - whether root or temp_dir are absolute path or not.
            #
            # See https://github.com/dcs4cop/xcube/issues/828
            #
            # Workaround used here is to manually download the directory.

            def get_files(source: str, target: str):
                source_items = fs.listdir(source, detail=True)
                for source_item in source_items:
                    source_path = source_item["name"]
                    source_type = source_item["type"]
                    source_name = source_path.replace("\\", "/").split("/")[-1]
                    target_path = os.path.join(target, source_name)
                    if source_type == "file":
                        fs.get_file(source_path, target_path)
                    elif source_type == "directory":
                        os.mkdir(target_path)
                        get_files(source_path, target_path)

            get_files(root, temp_dir)

            return FileSet(
                temp_dir,
                sub_path=self.sub_path,
                includes=self.includes,
                excludes=self.excludes,
            )
        else:
            _, temp_file = new_temp_file(
                prefix=TEMP_FILE_PREFIX, suffix=temp_file_suffix
            )
            fs.get_file(root, temp_file)
            return FileSet(
                temp_file,
                sub_path=self.sub_path,
                includes=self.includes,
                excludes=self.excludes,
            )

    def is_local_dir(self) -> bool:
        """Test whether this file set refers to an existing local directory."""
        local_path = self.get_local_path()
        return os.path.isdir(local_path) if local_path is not None else False

    def to_local_dir(self, dir_path: str = None) -> "FileSet":
        """Convert this file set into a file set that refers to a
        directory in the local file system.

        The *dir_path* parameter is used only if this fileset
        does not already refer to an existing local directory.

        Args:
            dir_path: An optional directory path. If not given, a
                temporary directory is created.

        Returns:
            The file set representing the local directory.
        """
        file_set = self.to_local()
        if (
            file_set.is_local_dir()
            and dir_path is None
            and not file_set.includes
            and not file_set.excludes
        ):
            return file_set
        return file_set._write_local_dir(dir_path)

    def is_local_zip(self):
        """Test whether this file set refers to an existing local ZIP archive."""
        local_path = self.get_local_path()
        return zipfile.is_zipfile(local_path) if local_path is not None else False

    def to_local_zip(self, zip_path: str = None) -> "FileSet":
        """Zip this file set and return it as a new local directory file set.
        If this is already a ZIP archive, return this file set.

        Args:
            zip_path: An optional path for the new ZIP archive. If not
                given, a temporary file will be created.

        Returns:
            The file set representing the local ZIP archive.
        """
        file_set = self.to_local()
        if (
            file_set.is_local_zip()
            and zip_path is None
            and not file_set._includes
            and not file_set._excludes
        ):
            return file_set
        return file_set._write_local_zip(zip_path)

    def _write_local_dir(self, dir_path: Optional[str]) -> "FileSet":
        """Write file set into local directory."""
        if dir_path:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            else:
                # We should assert an empty existing directory
                pass
        else:
            dir_path = new_temp_dir(prefix=TEMP_FILE_PREFIX)

        mapper = self._get_details().mapper

        sub_path = _normalize_sub_path(self.sub_path)

        for key in self.keys():
            file_path = os.path.join(dir_path, _key_to_path(key, sub_path))
            file_dir_path = os.path.dirname(file_path)
            if not os.path.exists(file_dir_path):
                os.makedirs(file_dir_path, exist_ok=True)
            with open(file_path, "wb") as stream:
                stream.write(mapper[key])

        return FileSet(dir_path)

    def _write_local_zip(self, zip_path: Optional[str]) -> "FileSet":
        """Write file set into local ZIP archive."""
        local_dir_path = self.get_local_path()

        if not zip_path:
            _, zip_path = new_temp_file(prefix=TEMP_FILE_PREFIX, suffix=".zip")

        sub_path = _normalize_sub_path(self.sub_path)

        with zipfile.ZipFile(zip_path, "w") as zip_file:
            for key in self.keys():
                file_path = os.path.join(local_dir_path, key)
                zip_file.write(
                    file_path, arcname=_strip_sub_path_from_key(key, sub_path)
                )

        return FileSet(zip_path, sub_path=self.sub_path)

    def _accepts_key(self, key: str) -> bool:
        if key == "":
            return False
        key = "/" + key.replace(os.path.sep, "/")
        return _key_matches(key, self._include_patterns, True) and not _key_matches(
            key, self._exclude_patterns, False
        )

    def _get_details(self) -> _FileSetDetails:
        if self._details is None:
            self._details = _FileSetDetails.new(self.path, self.storage_params)
        return self._details

    def _get_fs(self) -> fsspec.AbstractFileSystem:
        return self._get_details().fs


def _key_matches(key: str, patterns: list[re.Pattern], empty_value: bool) -> bool:
    if not patterns:
        return empty_value
    for pattern in patterns:
        if pattern.match(key):
            return True
    return False


def _translate_patterns(patterns: Optional[Collection[str]]) -> list[re.Pattern]:
    return [
        re.compile(fnmatch.translate(np))
        for p in patterns
        for np in _normalize_pattern(p)
    ]


def _normalize_sub_path(sub_path: str) -> str:
    if sub_path:
        while sub_path.startswith("/"):
            sub_path = sub_path[1:]
        while sub_path.endswith("/"):
            sub_path = sub_path[:-1]
        if sub_path:
            sub_path += "/"
    return sub_path or ""


def _normalize_pattern(pattern: str) -> list[str]:
    pattern = pattern.replace(os.path.sep, "/")
    start_sep = pattern.startswith("/")
    end_sep = pattern.endswith("/")
    if start_sep and end_sep:
        return [pattern, f"{pattern}*"]
    elif start_sep:
        return [pattern, f"{pattern}/*"]
    elif end_sep:
        return [pattern, f"{pattern}*", f"*/{pattern}", f"*/{pattern}*"]
    else:
        return [pattern, f"*/{pattern}", f"*/{pattern}/*"]


def _strip_sub_path_from_key(key: str, sub_path: str):
    if key.startswith(sub_path):
        return key[len(sub_path) :]
    return key


def _key_to_path(key: str, sub_path: str):
    key_path = key
    if sub_path:
        key_path = _strip_sub_path_from_key(key, sub_path)
    return key_path.replace("/", os.path.sep)
