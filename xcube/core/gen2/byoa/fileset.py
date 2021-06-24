# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import fnmatch
import os.path
import re
import zipfile
from typing import Any, Dict, Optional, List, Collection, Iterator, Tuple

import fsspec

from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonObject
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema
from .temp import new_temp_dir
from .temp import new_temp_file

# Following set may need some refinement.
# There should be some implementation in fsspec.
_REMOTE_PROTOCOLS = {
    'adl', 'abfs', 'az',  # Azure File Systems
    'dask',  # Dask worker file system
    'ftp',  # FTP
    'gcs', 'gs',  # Google Cloud Storage
    'github',  # GitHub
    'hdfs',  # Hadoop Distributed File System
    'https', 'http',  # HTTP
    's3',  # AWS S3-like
}


class FileSet(JsonObject):
    """
    A set of files that can found at some abstract root *path*.
    The *path* may identify local or remote filesystems.
    A filesystem may require specific *parameters*, e.g.
    user credentials.

    This implementation is based on the ```fsspec```
    package and the documentation about the different
    possible file systems and their specific parameters
    can be found at https://filesystem-spec.readthedocs.io/.

    Examples:

        FileSet('eurodatacube/my_processor')

        FileSet('eurodatacube/my_processor.zip')

        FileSet('s3://eurodatacube/my_processor.zip',
                parameters=dict(key='...',
                                secret='...')

        FileSet('github://dcs4cop:xcube@v0.8.1/',
                parameters=dict(username='...',
                                token='ghp_...')

    If *includes* wildcard patterns are given, only files
    whose paths match any of the patterns are included.
    If *excludes* wildcard patterns are given, only files
    whose paths do not match all of the patterns are included.

    :param path: Root path of the file set.
    :param includes: Wildcard patterns used to include a file.
    :param excludes: Wildcard patterns used to exclude a file.
    :param parameters: File system specific parameters.
    """

    def __init__(self,
                 path: str,
                 includes: Collection[str] = None,
                 excludes: Collection[str] = None,
                 parameters: Dict[str, Any] = None):
        assert_instance(path, str, 'path')
        assert_given(path, 'path')
        self._path = path
        self._parameters = dict(parameters) if parameters is not None else None
        self._includes = list(includes) if includes is not None else None
        self._excludes = list(excludes) if excludes is not None else None
        # computed members
        self._include_patterns = _translate_patterns(includes or [])
        self._exclude_patterns = _translate_patterns(excludes or [])
        # cached, computed members
        self._fs_root: Optional[Tuple[fsspec.AbstractFileSystem, str]] = None
        self._mapper: Optional[fsspec.FSMap] = None

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        """Get the JSON-schema for FileSet objects."""
        return JsonObjectSchema(
            properties=dict(
                path=JsonStringSchema(min_length=1),
                parameters=JsonObjectSchema(additional_properties=True),
                includes=JsonArraySchema(items=JsonStringSchema(min_length=1)),
                excludes=JsonArraySchema(items=JsonStringSchema(min_length=1)),
            ),
            additional_properties=False,
            required=['path'],
            factory=cls,
        )

    @property
    def path(self) -> str:
        """Get the root path."""
        return self._path

    @property
    def parameters(self) -> Optional[Dict[str, Any]]:
        """
        Get optional parameters for the file system
        :attribute:path is referring to.
        """
        return self._parameters

    @property
    def includes(self) -> Optional[List[str]]:
        """Wildcard patterns used to include a file."""
        return self._includes

    @property
    def excludes(self) -> Optional[List[str]]:
        """Wildcard patterns used to exclude a file."""
        return self._excludes

    def keys(self) -> Iterator[str]:
        """
        Get keys in this file set.
        A key is a normalized path relative to the root *path*.
        The forward slash "/" is used as path separator.
        """
        for key in self._get_mapper().keys():
            # print('-->', key)
            if self._accepts_key(key):
                yield key

    def is_remote(self) -> bool:
        """
        Test whether this file set refers to a remote location.
        """
        urls = self._path.split('::')
        count = len(urls)
        for url, index in zip(urls, range(count)):
            if index < count - 1 and url in _REMOTE_PROTOCOLS:
                return True
            protocol_and_path = url.split('://', maxsplit=1)
            if len(protocol_and_path) == 2:
                if protocol_and_path[0] in _REMOTE_PROTOCOLS:
                    return True
        return False

    def is_local_dir(self) -> bool:
        """
        Test whether this file set refers to an existing local directory.
        """
        path = _to_local_path(self.path)
        if path is None:
            return False
        return os.path.isdir(path)

    def to_local_dir(self, dir_path: str = None) -> 'FileSet':
        """
        Convert this file set into a file set that refers to a
        directory in the local file system.

        :param dir_path: An optional directory path.
            If not given, a temporary directory is created.
        :return: The file set representing the local directory.
        """
        path = _to_local_path(self.path)

        if path is not None and os.path.isdir(path):
            # ignore given dir_path
            return self

        if dir_path:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            else:
                # We should assert an empty existing directory
                pass
        else:
            dir_path = new_temp_dir()

        mapper = self._get_mapper()
        for key in self.keys():
            file_path = os.path.join(dir_path, key.replace('/', os.path.sep))
            file_dir_path = os.path.dirname(file_path)
            if not os.path.isdir(file_dir_path):
                os.makedirs(file_dir_path)
            with open(file_path, 'wb') as stream:
                stream.write(mapper[key])

        return FileSet(dir_path)

    def is_local_zip(self):
        """
        Test whether this file set refers to an existing local ZIP archive.
        """
        path = _to_local_path(self.path)
        if path is None:
            return False
        return zipfile.is_zipfile(path)

    def to_local_zip(self, zip_path: str = None) -> 'FileSet':
        """
        Zip this file set and return it as a new local directory file set.
        If this is already a ZIP archive, return this file set.

        :param zip_path: An optional path for the new ZIP archive.
            If not given, a temporary file will be created.
        :return: The file set representing the local ZIP archive.
        """
        if self.is_local_zip():
            # ignore given zip_path
            return self
        if not zip_path:
            zip_path = new_temp_file(suffix='.zip')
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            for key in self.keys():
                file_path = os.path.join(self._path, key)
                zip_file.write(file_path, arcname=key)
        return FileSet(zip_path)

    def _accepts_key(self, key: str) -> bool:
        key = '/' + key.replace(os.path.sep, "/")
        return _key_matches(key, self._include_patterns, True) \
               and not _key_matches(key, self._exclude_patterns, False)

    def _get_mapper(self) -> fsspec.FSMap:
        if self._mapper is None:
            fs, root = self._get_fs_root()
            self._mapper = fsspec.FSMap(root, fs)
        return self._mapper

    def _get_fs_root(self) -> Tuple[fsspec.AbstractFileSystem, str]:
        if self._fs_root is None:
            url_path = self._path
            if '://' not in url_path:
                url_path = 'file://' + url_path
            if zipfile.is_zipfile(self._path) or self._path.endswith('.zip'):
                url_path = 'zip::' + url_path
            self._fs_root = fsspec.core.url_to_fs(url_path,
                                                  **(self._parameters or {}))
        return self._fs_root

    def _get_fs(self) -> fsspec.AbstractFileSystem:
        return self._get_fs_root()[0]


def _to_local_path(path: str) -> Optional[str]:
    url = path.split('::')[-1]
    if '://' not in url:
        return url
    if url.startswith('file://'):
        return url[len('file://'):]
    return None


def _key_matches(key: str,
                 patterns: List[re.Pattern],
                 empty_value: bool) -> bool:
    if not patterns:
        return empty_value
    for pattern in patterns:
        if pattern.match(key):
            return True
    return False


def _translate_patterns(patterns: Optional[Collection[str]]) \
        -> List[re.Pattern]:
    return [re.compile(fnmatch.translate(np))
            for p in patterns
            for np in _normalize_pattern(p)]


def _normalize_pattern(pattern: str) -> List[str]:
    pattern = pattern.replace(os.path.sep, '/')
    start_sep = pattern.startswith('/')
    end_sep = pattern.endswith('/')
    if start_sep and end_sep:
        return [pattern,
                f'{pattern}*']
    elif start_sep:
        return [pattern,
                f'{pattern}/*']
    elif end_sep:
        return [pattern,
                f'{pattern}*',
                f'*/{pattern}',
                f'*/{pattern}*']
    else:
        return [pattern,
                f'*/{pattern}',
                f'*/{pattern}/*']
