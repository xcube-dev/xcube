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
import tempfile
import zipfile
from typing import Any, Dict, Optional, List, Collection, Iterator, Tuple

import fsspec

from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonObject
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema
from xcube.util.jsonschema import JsonArraySchema
from .constants import DEFAULT_TEMP_FILE_PREFIX


class FileSet(JsonObject):

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
        self._include_patterns = self._translate_patterns(includes or [])
        self._exclude_patterns = self._translate_patterns(excludes or [])
        # cached, computed members
        self._fs_root: Optional[Tuple[fsspec.AbstractFileSystem, str]] = None
        self._mapper: Optional[fsspec.FSMap] = None

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
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
        return self._path

    @property
    def parameters(self) -> Optional[Dict[str, Any]]:
        return self._parameters

    @property
    def includes(self) -> Optional[List[str]]:
        return self._includes

    @property
    def excludes(self) -> Optional[List[str]]:
        return self._excludes

    def keys(self) -> Iterator[str]:
        for key in self._get_mapper().keys():
            if self._accepts_key(key):
                yield key

    def is_local_dir(self):
        return os.path.isdir(self._path)

    def to_local_dir(self, dir_path: str = None) -> 'FileSet':

        if os.path.isdir(self._path):
            # ignore given dir_path
            return self

        if dir_path:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        else:
            dir_path = tempfile.mkdtemp(prefix=DEFAULT_TEMP_FILE_PREFIX)

        mapper = self._get_mapper()
        for key in self.keys():
            file_path = os.path.join(dir_path, key)
            file_dir_path = os.path.dirname(file_path)
            if not os.path.isdir(file_dir_path):
                os.mkdir(file_dir_path)
            with open(file_path, 'wb') as stream:
                stream.write(mapper[key])

        return FileSet(dir_path)

    def is_local_zip(self):
        return zipfile.is_zipfile(self._path)

    def to_local_zip(self, zip_path: str = None) -> 'FileSet':
        """
        Zip this file set and return it as a new local directory file set.
        If this is already a ZIP archive, return this file set.

        :param zip_path: The desired local ZIP archive path.
        :return: the file set representing the ZIP archive.
        """
        if self.is_local_zip():
            # ignore given zip_path
            return self
        if not zip_path:
            zip_path = tempfile.mktemp(prefix=DEFAULT_TEMP_FILE_PREFIX, suffix='.zip')
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            for key in self.keys():
                file_path = os.path.join(self._path, key)
                zip_file.write(file_path, arcname=key)
        return FileSet(zip_path)

    def _accepts_key(self, key: str) -> bool:
        key = '/' + key.replace(os.path.sep, "/")
        return self._key_matches(key, self._include_patterns, True) \
               and not self._key_matches(key, self._exclude_patterns, False)

    @classmethod
    def _key_matches(cls, key: str, patterns: List[re.Pattern], empty_value: bool) -> bool:
        if not patterns:
            return empty_value
        for pattern in patterns:
            if pattern.match(key):
                return True
        return False

    @classmethod
    def _translate_patterns(cls, patterns: Optional[Collection[str]]) -> List[re.Pattern]:
        return [re.compile(fnmatch.translate(np))
                for p in patterns
                for np in cls._normalize_pattern(p)]

    @classmethod
    def _normalize_pattern(cls, pattern: str) -> List[str]:
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
