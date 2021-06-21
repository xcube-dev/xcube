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

import inspect
import warnings
from typing import Any, Union, Dict, Callable

from xcube.util.assertions import assert_condition
from xcube.util.assertions import assert_given
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonObject
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema
from .fileset import FileSet


class CodeConfig(JsonObject):

    def __init__(self,
                 callable_name: str = None,
                 parameters: Dict[str, Any] = None,
                 code_string: str = None,
                 file_set: FileSet = None,
                 zip_file: str = None,
                 gh_user: str = None,
                 gh_repo: str = None,
                 gh_tag: str = None,
                 gh_token: str = None,
                 install_package: bool = False):
        assert_given(callable_name, 'callable_name')
        if any((gh_user, gh_repo, gh_tag, gh_token)):
            assert_condition(all((gh_user, gh_repo, gh_tag)),
                             'if any of gh_user, gh_repo, gh_tag, gh_token are given, '
                             'then gh_user, gh_repo, gh_tag must be given too')
        self.callable_name = callable_name
        self.parameters = parameters
        self.code_string = code_string
        self.file_set = file_set
        self.zip_file = zip_file
        self.gh_user = gh_user
        self.gh_repo = gh_repo
        self.gh_tag = gh_tag
        self.gh_token = gh_token
        self.install_package = install_package

    def for_service(self):
        if self.file_set is not None:
            return ServiceCodeConfig.from_file_set(self.file_set,
                                                   callable_name=self.callable_name,
                                                   parameters=self.parameters,
                                                   install_package=self.install_package)
        return self

    @classmethod
    def from_code(cls,
                  *code: Union[str, Callable],
                  callable_name: str = None,
                  parameters: Dict[str, Any] = None) -> 'CodeConfig':
        first_callable = next(filter(callable, code), None)
        if first_callable is not None and not callable_name:
            callable_name = first_callable.__name__
        if not callable_name:
            callable_name = 'process_dataset'
        if all(map(callable, code)):
            warnings.warn('source code may be missing required import statements')
        code_string = '\n\n'.join([c if isinstance(c, str)
                                   else inspect.getsource(c)
                                   for c in code])
        return CodeConfig(callable_name=callable_name,
                          parameters=parameters,
                          code_string=code_string)

    @classmethod
    def from_zip(cls,
                 zip_file: str,
                 callable_name: str,
                 parameters: Dict[str, Any] = None,
                 install_package: bool = False) -> 'CodeConfig':
        assert_given(zip_file, 'zip_file')
        assert_given(callable_name, 'callable_name')
        return CodeConfig(callable_name=callable_name,
                          parameters=parameters,
                          zip_file=zip_file,
                          install_package=install_package)

    @classmethod
    def from_file_set(cls,
                      file_set: FileSet,
                      callable_name: str,
                      parameters: Dict[str, Any] = None,
                      install_package: bool = False) -> 'CodeConfig':
        assert_given(file_set, 'file_set')
        assert_given(callable_name, 'callable_name')
        return CodeConfig(callable_name=callable_name,
                          parameters=parameters,
                          file_set=file_set,
                          install_package=install_package)

    @classmethod
    def from_github(cls,
                    callable_name: str,
                    gh_user: str,
                    gh_repo: str,
                    gh_tag: str,
                    gh_token: str = None,
                    parameters: Dict[str, Any] = None,
                    install_package: bool = False) -> 'CodeConfig':
        assert_given(gh_user, 'gh_user')
        assert_given(gh_repo, 'gh_repo')
        assert_given(gh_tag, 'gh_tag')
        return CodeConfig(callable_name=callable_name,
                          parameters=parameters,
                          gh_user=gh_user,
                          gh_repo=gh_repo,
                          gh_tag=gh_tag,
                          gh_token=gh_token,
                          install_package=install_package)

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                callable_name=JsonStringSchema(min_length=1),
                parameters=JsonObjectSchema(additional_properties=True),
                code_string=JsonStringSchema(min_length=1),
                files=JsonArraySchema(JsonStringSchema(min_length=1)),
                zip_file=JsonStringSchema(min_length=1),
                gh_user=JsonStringSchema(min_length=1),
                gh_repo=JsonStringSchema(min_length=1),
                gh_tag=JsonStringSchema(min_length=1),
                gh_token=JsonStringSchema(min_length=1),
                install_package=JsonBooleanSchema(min_length=1),
            ),
            additional_properties=False,
            required=['callable_name'],
            factory=cls,
        )


class ServiceCodeConfig(CodeConfig):

    @classmethod
    def from_file_set(cls,
                      file_set: FileSet,
                      callable_name: str = None,
                      parameters: Dict[str, Any] = None,
                      install_package: bool = False) -> 'CodeConfig':
        assert_given(file_set, 'file_set')
        assert_given(callable_name, 'callable_name')
        zip_file = file_set.zip()
        return cls.from_zip(callable_name=callable_name,
                            parameters=parameters,
                            zip_file=zip_file,
                            install_package=install_package)
