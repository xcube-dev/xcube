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
from typing import Any, Dict, Optional, List, Collection, Iterator, Union

import fsspec

from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonObject
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema
from .temp import new_temp_dir
from .temp import new_temp_file


class _FileSetDetails:
    def __init__(self,
                 fs: fsspec.AbstractFileSystem,
                 root: str,
                 local_path: Union[None, str]):
        """
        Stores file system details about a FileSet instance.
        Internal helper class.
        :param fs: file system
        :param root: root in file system
        :param local_path: the local or network path.
            None, if this is not a local path.
        """
        self._fs = fs
        self._root = root
        self._local_path = local_path
        self._mapper: Optional[fsspec.FSMap] = None

    @classmethod
    def new(cls,
            path: str,
            parameters: Dict[str, Any] = None) -> '_FileSetDetails':
        try:
            fs, root = fsspec.core.url_to_fs(path,
                                             **(parameters or {}))
        except (ImportError, OSError) as e:
            raise ValueError(f'Illegal file set {path!r}') from e
        local_path = root if fs.protocol == 'file' else None
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
            if self.local_path \
                    and zipfile.is_zipfile(self.local_path):
                from fsspec.implementations.zip import ZipFileSystem
                fs, root = ZipFileSystem(self.local_path), ''
            self._mapper = fsspec.FSMap(root, fs)
        return self._mapper


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
        self._details: Optional[_FileSetDetails] = None

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
        for key in self._get_details().mapper.keys():
            # print('-->', key)
            if self._accepts_key(key):
                yield key

    def is_remote(self) -> bool:
        """
        Test whether this file set refers to a remote location.
        """
        return self._get_details().local_path is None

    def is_local(self) -> bool:
        """
        Test whether this file set refers to a local file or directory.
        """
        return self._get_details().local_path is not None

    def is_local_dir(self) -> bool:
        """
        Test whether this file set refers to an existing local directory.
        """
        local_path = self._get_details().local_path
        return os.path.isdir(local_path) \
            if local_path is not None else False

    def to_local_dir(self, dir_path: str = None) -> 'FileSet':
        """
        Convert this file set into a file set that refers to a
        directory in the local file system.

        The *dir_path* parameter is used only if this fileset
        does not already refer to an existing local directory.

        :param dir_path: An optional directory path.
            If not given, a temporary directory is created.
        :return: The file set representing the local directory.
        """
        if self.is_local_dir() \
                and dir_path is None \
                and not self.includes \
                and not self.excludes:
            return self

        if self.is_remote():
            return FileSet('simplecache::' + self.path,
                           includes=self.includes,
                           excludes=self.excludes,
                           parameters=self.parameters) \
                .to_local_dir(dir_path=dir_path)

        if dir_path:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            else:
                # We should assert an empty existing directory
                pass
        else:
            dir_path = new_temp_dir()

        mapper = self._get_details().mapper
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
        local_path = self._get_details().local_path
        return zipfile.is_zipfile(local_path) \
            if local_path is not None else False

    def to_local_zip(self, zip_path: str = None) -> 'FileSet':
        """
        Zip this file set and return it as a new local directory file set.
        If this is already a ZIP archive, return this file set.

        :param zip_path: An optional path for the new ZIP archive.
            If not given, a temporary file will be created.
        :return: The file set representing the local ZIP archive.
        """
        if self.is_local_zip() \
                and zip_path is None \
                and not self._includes \
                and not self._excludes:
            return self

        if self.is_remote():
            return FileSet('simplecache::' + self.path,
                           includes=self.includes,
                           excludes=self.excludes,
                           parameters=self.parameters) \
                .to_local_zip(zip_path=zip_path)

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

    def _get_details(self) -> _FileSetDetails:
        if self._details is None:
            self._details = _FileSetDetails.new(self.path, self.parameters)
        return self._details

    def _get_fs(self) -> fsspec.AbstractFileSystem:
        return self._get_details().fs


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
