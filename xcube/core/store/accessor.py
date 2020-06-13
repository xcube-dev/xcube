# The MIT License (MIT)
# Copyright (c) 2020 by the xcube development team and contributors
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

from abc import abstractmethod, ABC
from typing import Any, List, Optional

import xarray as xr

from xcube.constants import EXTENSION_POINT_DATA_OPENERS
from xcube.constants import EXTENSION_POINT_DATA_WRITERS
from xcube.util.assertions import assert_given
from xcube.util.extension import Extension
from xcube.util.extension import ExtensionPredicate
from xcube.util.extension import ExtensionRegistry
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.plugin import get_extension_registry


#######################################################
# Data accessor instantiation and registry query
#######################################################

def new_data_opener(opener_id: str,
                    extension_registry: Optional[ExtensionRegistry] = None) -> 'DataOpener':
    """
    Get an instance of the data opener identified by *opener_id*.

    :param opener_id: The data opener identifier.
    :param extension_registry: Optional extension registry. If not given, the global extension registry will be used.
    :return: A data opener instance.
    """
    assert_given(opener_id, 'opener_id')
    extension_registry = extension_registry or get_extension_registry()
    return extension_registry.get_component(EXTENSION_POINT_DATA_OPENERS, opener_id)()


def new_data_writer(writer_id: str,
                    extension_registry: Optional[ExtensionRegistry] = None) -> 'DataWriter':
    """
    Get an instance of the data opener identified by *writer_id*.

    :param extension_registry: Optional extension registry. If not given, the global extension registry will be used.
    :return: A data writer instance.
    """
    assert_given(writer_id, 'writer_id')
    extension_registry = extension_registry or get_extension_registry()
    return extension_registry.get_component(EXTENSION_POINT_DATA_WRITERS, writer_id)()


def find_data_opener_extensions(predicate: ExtensionPredicate = None,
                                extension_registry: Optional[ExtensionRegistry] = None) -> List[Extension]:
    """
    Get registered data opener extensions using the optional filter function *predicate*.

    :param predicate: An optional filter function.
    :param extension_registry: Optional extension registry. If not given, the global extension registry will be used.
    :return: List of matching extensions.
    """
    extension_registry = extension_registry or get_extension_registry()
    return extension_registry.find_extensions(EXTENSION_POINT_DATA_OPENERS, predicate=predicate)


def find_data_writer_extensions(predicate: ExtensionPredicate = None,
                                extension_registry: Optional[ExtensionRegistry] = None) -> List[Extension]:
    """
    Get registered data writer extensions using the optional filter function *predicate*.

    :param predicate: An optional filter function.
    :param extension_registry: Optional extension registry. If not given, the global extension registry will be used.
    :return: List of matching extensions.
    """
    extension_registry = extension_registry or get_extension_registry()
    return extension_registry.find_extensions(EXTENSION_POINT_DATA_WRITERS, predicate=predicate)


def get_data_accessor_predicate(type_id: str = None,
                                format_id: str = None,
                                storage_id: str = None) -> ExtensionPredicate:
    """
    Get a predicate that checks if a data accessor extensions's name is
    compliant with *type_id*, *format_id*, *storage_id*.

    :param type_id: Optional data type identifier to be supported.
    :param format_id: Optional data format identifier to be supported.
    :param storage_id: Optional data storage identifier to be supported.
    :return: A filter function.
    """
    if any((type_id, format_id, storage_id)):
        def predicate(extension: Extension) -> bool:
            parts = extension.name.split(':', maxsplit=4)
            if len(parts) < 3:
                raise DataAccessorError(f'Illegal data opener/writer extension name "{extension.name}"')
            type_ok = type_id is None or parts[0] == '*' or parts[0] == type_id
            format_ok = format_id is None or parts[1] == '*' or parts[1] == format_id
            storage_ok = storage_id is None or parts[2] == '*' or parts[2] == storage_id
            return type_ok and format_ok and storage_ok
    else:
        # noinspection PyUnusedLocal
        def predicate(extension: Extension) -> bool:
            return True

    return predicate


#######################################################
# Classes
#######################################################


class DataOpener(ABC):
    @abstractmethod
    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        """
        Get the schema for the parameters passed as *open_params* to :meth:open_data(data_id, open_params).
        If *data_id* is given, the returned schema will be tailored to the constraints implied by the
        identified data resource. Some openers might not support this, therefore *data_id* is optional, and if
        it is omitted, the returned schema will be less restrictive.

        :param data_id: An optional data resource identifier.
        :return: The schema for the parameters in *open_params*.
        """

    @abstractmethod
    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        """
        Open the data resource given by the data resource identifier *data_id* using the supplied *open_params*.

        :param data_id: The data resource identifier.
        :param open_params: Opener-specific parameters.
        :return: An xarray.Dataset instance.
        """


class DataDeleter(ABC):
    @abstractmethod
    def delete_data(self, data_id: str):
        """
        Delete a data resource.

        :param data_id: A data resource identifier known to exist.
        """


class DataWriter(DataDeleter, ABC):
    @abstractmethod
    def get_write_data_params_schema(self) -> JsonObjectSchema:
        """
        Get the schema for the parameters passed as *write_params* to
        :meth:write_data(data resource, data_id, open_params).

        :return: The schema for the parameters in *write_params*.
        """

    @abstractmethod
    def write_data(self, data: Any, data_id: str, replace: bool = False, **write_params) -> str:
        """
        Write a data resource using the supplied *data_id* and *write_params*.

        :param data: The data resource's in-memory representation to be written.
        :param data_id: A unique data resource identifier.
        :param replace: Whether to replace an existing data resource.
        :param write_params: Writer-specific parameters.
        :return: The data resource identifier used to write the data resource.
        """


class DataTimeSliceUpdater(DataWriter, ABC):
    @abstractmethod
    def append_data_time_slice(self, data_id: str, time_slice: xr.Dataset):
        """
        Append a time slice to the identified data resource.

        :param data_id: The data resource identifier.
        :param time_slice: The time slice data to be inserted. Must be compatible with the data resource.
        """

    @abstractmethod
    def insert_data_time_slice(self, data_id: str, time_slice: Any, time_index: int):
        """
        Insert a time slice into the identified data resource at given index.

        :param data_id: The data resource identifier.
        :param time_slice: The time slice data to be inserted. Must be compatible with the data resource.
        :param time_index: The time index.
        """

    @abstractmethod
    def replace_data_time_slice(self, data_id: str, time_slice: Any, time_index: int):
        """
        Replace a time slice in the identified data resource at given index.

        :param data_id: The data resource identifier.
        :param time_slice: The time slice data to be inserted. Must be compatible with the data resource.
        :param time_index: The time index.
        """


class DataAccessorError(Exception):
    """
    Raised on error in any of the data accessor methods.

    :param message: The error message.
    """

    def __init__(self, message: str):
        super().__init__(message)
