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

import importlib
import inspect
import os.path
import sys
import tempfile
import warnings
from typing import Any, Union, Dict, Callable, Optional, Sequence

from xcube.util.assertions import assert_condition
from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonObject
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema
from .fileset import FileSet


class CodeConfig(JsonObject):
    """
    Code configuration object.

    Instances should always be created using one of the factory methods:

    * :meth:from_code
    * :meth:from_callable
    * :meth:from_file_set

    :param callable_ref: Reference to the callable in the *file_set*,
        must have form "<module-name>:<callable-name>"
    :param parameters: The parameters passed
        as keyword-arguments to the callable.
    :param inline_code: An inline source code string.
        Cannot be used if *file_set* is given.
    :param file_set: A file set that contains Python
        modules or packages.
        Cannot be used if *inline_code* is given.
    :param install_required: Whether *file_set* contains
        Python modules or packages that must be installed.
    """

    def __init__(self,
                 callable_ref: str = None,
                 parameters: Dict[str, Any] = None,
                 inline_code: str = None,
                 file_set: FileSet = None,
                 install_required: bool = None):
        assert_given(callable_ref, 'callable_ref')
        if inline_code and file_set:
            assert_condition(False,
                             'only one of inline_code and file_set can be given')
        elif inline_code:
            assert_condition(file_set is None,
                             'only one of inline_code and file_set can be given')
        elif file_set:
            assert_condition(inline_code is None,
                             'only one of inline_code and file_set can be given')
        else:
            assert_condition(False,
                             'one of inline_code or file_set must be given')
        self.callable_ref = callable_ref
        self.parameters = parameters
        self.inline_code = inline_code
        self.file_set = file_set
        self.install_required = install_required
        self._callable: Optional[Callable] = None

    def for_service(self) -> 'CodeConfig':
        """
        Convert this code configuration so can be used by the generator service.
        If this code configuration uses a file set the file set is a local directory,
        zip it and return a new local ZIP archive file set.
        Otherwise return this configuration as-is.
        """
        if self.file_set is not None \
                and self.file_set.is_local_dir():
            return self.from_file_set(file_set=self.file_set.to_local_zip(),
                                      callable_ref=self.callable_ref,
                                      parameters=self.parameters,
                                      install_required=self.install_required)
        return self

    @classmethod
    def from_code(cls,
                  code: Union[str, Callable, Sequence[Union[str, Callable]]],
                  callable_name: str = None,
                  module_name: str = None,
                  parameters: Dict[str, Any] = None) -> 'CodeConfig':
        """
        Create a code configuration from the given *code* which may be
        a code string or a callable or a sequence of code strings
        or a callables.

        This will create a configuration that uses an inline
        ``code_string`` which contains the source code.

        :param code: The code.
        :param callable_name: The callable name.
            If not given, will be inferred from first callable.
            Otherwise it defaults to "process_dataset".
        :param module_name: The module name. If not given,
            defaults to "user_code".
        :param parameters: The parameters passed
            as keyword-arguments to the callable.
        :return: A new code configuration.
        """
        assert_given(code, 'code')
        if isinstance(code, str) or isinstance(code, Callable):
            code = [code]
        if not callable_name:
            first_callable = next(filter(callable, code), None)
            if first_callable is not None:
                callable_name = first_callable.__name__
        callable_name = callable_name or 'process_dataset'
        module_name = module_name or _next_user_module_name()
        if all(map(callable, code)):
            warnings.warn('source code may be missing '
                          'import statements to be executable')
        code_string = '\n\n'.join([c if isinstance(c, str)
                                   else inspect.getsource(c)
                                   for c in code])
        return CodeConfig(callable_ref=f'{module_name}:{callable_name}',
                          parameters=parameters,
                          inline_code=code_string)

    @classmethod
    def from_callable(cls,
                      func_or_class: Callable,
                      parameters: Dict[str, Any] = None) -> 'CodeConfig':
        """
        Create a code configuration from the callable *func_or_class*.

        This will create a configuration that uses a ``file_set``
        which contains the source code for the *func_or_class*.

        :param func_or_class: A function or class
        :param parameters: The parameters passed
            as keyword-arguments to the callable.
        :return: A new code configuration.
        """

        callable_name = func_or_class.__name__
        if not callable_name:
            raise ValueError(f'cannot detect name '
                             f'for func_or_class')

        module = inspect.getmodule(func_or_class)
        module_name = module.__name__ if module is not None else None
        if not module_name:
            raise ValueError(f'cannot detect module '
                             f'for callable {callable_name!r}')

        source_file = inspect.getabsfile(func_or_class)
        if source_file is None:
            raise ValueError(f'cannot detect source file '
                             f'for func_or_class {callable_name!r}')

        module_path, ext = os.path.splitext(os.path.normpath(source_file))
        if not module_path.replace(os.path.sep, '/') \
                .endswith(module_name.replace('.', '/')):
            raise ValueError(f'cannot detect module path '
                             f'for func_or_class {callable_name!r}')

        module_path = os.path.normpath(module_path[0: -len(module_name)])

        code_config = CodeConfig(callable_ref=f'{module_name}:{callable_name}',
                                 file_set=FileSet(module_path, includes=['*.py']),
                                 parameters=parameters)
        code_config.set_callable(func_or_class)
        return code_config

    @classmethod
    def from_file_set(cls,
                      file_set: Union[str, FileSet],
                      callable_ref: str,
                      parameters: Dict[str, Any] = None,
                      install_required: bool = False) -> 'CodeConfig':
        """
        Create a code configuration from a file set.

        :param file_set: The file set.
        :param callable_ref: Reference to the callable in the *file_set*,
            must have form "<module-name>:<callable-name>"
        :param parameters: Parameters to be passed
            as keyword-arguments to the the callable.
        :param install_required: Whether the *file_set* is
            a package that must be installed.
        :return: A new code configuration.
        """
        if isinstance(file_set, str):
            file_set = FileSet(file_set)
        else:
            assert_instance(file_set, FileSet, 'file_set')
        assert_given(callable_ref, 'callable_ref')
        return CodeConfig(callable_ref=callable_ref,
                          parameters=parameters,
                          file_set=file_set,
                          install_required=install_required)

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        """Get the JSON schema for CodeConfig objects."""
        return JsonObjectSchema(
            properties=dict(
                callable_ref=JsonStringSchema(min_length=1),
                parameters=JsonObjectSchema(additional_properties=True),
                inline_code=JsonStringSchema(min_length=1),
                file_set=FileSet.get_schema(),
                install_required=JsonBooleanSchema(),
            ),
            additional_properties=False,
            required=['callable_ref'],
            factory=cls,
        )

    def get_callable(self) -> Callable:
        """
        Get the callable specified by this configuration.

        In the common case, this will require importing the callable.

        :return: A callable
        :raise ImportError if the callable can not be imported
        """
        if self._callable is None:
            self.set_callable(self.load_callable())
        return self._callable

    def set_callable(self, func_or_class: Callable):
        """
        Set the callable that is represented by this configuration.
        :param func_or_class: A callable
        """
        assert_condition(callable(func_or_class), f'func_or_class must be callable')
        self._callable = func_or_class

    def load_callable(self) -> Callable:
        """
        Load the callable specified by this configuration.

        :return: A callable
        :raise ImportError if the callable can not be imported
        """
        module_name, callable_name = self.callable_ref.split(':')
        if self.file_set:
            dir_path = self.file_set.to_local_dir().path
        elif self.inline_code:
            dir_path = tempfile.mkdtemp()
            with open(os.path.join(dir_path, f'{module_name}.py'), 'w') as stream:
                stream.write(self.inline_code)
        else:
            raise RuntimeError('CodeConfig object has an illegal state')
        if self.install_required:
            warnings.warn(f'This user code configuration requires package installation,'
                          f' but this is not supported yet')
        return _load_callable(dir_path, module_name, callable_name)


def _load_callable(dir_path: str, module_name: str, callable_name: str) -> Callable:
    print(dir_path)
    sys.path = [dir_path] + sys.path
    module = importlib.import_module(module_name)
    func_or_class = getattr(module, callable_name, None)
    if func_or_class is None:
        raise ImportError(f'callable {callable_name!r} '
                          f'not found in module {module_name!r}')
    if not callable(func_or_class):
        raise ImportError(f'{module_name}:{callable_name} '
                          f'is not callable')
    return func_or_class


_user_module_counter = 0


def _next_user_module_name() -> str:
    global _user_module_counter
    _user_module_counter += 1
    return f'usercode{_user_module_counter}'
