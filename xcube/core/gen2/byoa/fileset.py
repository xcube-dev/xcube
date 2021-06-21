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
from typing import Optional, Iterator, List, Collection

from xcube.util.assertions import assert_given
from xcube.util.jsonschema import JsonObject, JsonObjectSchema, JsonStringSchema, JsonArraySchema


class FileSet(JsonObject):

    def __init__(self,
                 base_dir: str,
                 includes: Collection[str] = None,
                 excludes: Collection[str] = None):
        assert_given(base_dir, 'base_dir')
        self._base_dir = os.path.normpath(base_dir)
        self._includes = list(includes) if includes is not None else includes
        self._excludes = list(excludes) if excludes is not None else excludes
        self._include_patterns = self._translate_patterns(includes or [])
        self._exclude_patterns = self._translate_patterns(excludes or [])

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                base_dir=JsonStringSchema(min_length=1),
                includes=JsonArraySchema(items=JsonStringSchema(min_length=1)),
                excludes=JsonArraySchema(items=JsonStringSchema(min_length=1)),
            ),
            additional_properties=False,
            required=['base_dir'],
            factory=cls,
        )

    @property
    def base_dir(self) -> str:
        return self._base_dir

    @property
    def includes(self) -> List[str]:
        return self._includes

    @property
    def excludes(self) -> List[str]:
        return self._excludes

    @property
    def files(self) -> Iterator[str]:
        for root, dirs, files in os.walk(self._base_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if self.includes_path(file_path) \
                        and not self.excludes_path(file_path):
                    yield file_path

    def includes_path(self, file_path: str) -> bool:
        return self._matches_path(self._include_patterns, True, file_path.replace(os.path.sep, "/"))

    def excludes_path(self, file_path: str) -> bool:
        return self._matches_path(self._exclude_patterns, False, file_path.replace(os.path.sep, "/"))

    def zip(self, zip_path: str = None) -> str:
        if not zip_path:
            zip_path = tempfile.mktemp(suffix='.zip')
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            for file_path in self.files:
                item_name = os.path.relpath(file_path, self._base_dir)
                zip_file.write(file_path, arcname=item_name)
        return zip_path

    @classmethod
    def _matches_path(cls,
                      patterns: List[re.Pattern],
                      empty_value: bool,
                      path: str) -> bool:
        if not patterns:
            return empty_value
        for exclude in patterns:
            if exclude.match(path):
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
                    f'/{pattern}',
                    f'/{pattern}/*',
                    f'*/{pattern}',
                    f'*/{pattern}/*']
