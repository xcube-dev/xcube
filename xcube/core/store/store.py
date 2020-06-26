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
from typing import Iterator, Tuple, Any, Optional, List, Type

from xcube.constants import EXTENSION_POINT_DATA_STORES
from xcube.util.extension import Extension
from xcube.util.extension import ExtensionPredicate
from xcube.util.extension import ExtensionRegistry
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.plugin import get_extension_registry
from .accessor import DataOpener
from .accessor import DataWriter
from .descriptor import DataDescriptor
from .error import DataStoreError


#######################################################
# Data store instantiation and registry query
#######################################################

def new_data_store(data_store_id: str,
                   extension_registry: Optional[ExtensionRegistry] = None,
                   **data_store_params) -> 'DataStore':
    """
    Create a new data store instance for given *data_store_id* and *data_store_params*.

    :param data_store_id: A data store identifier.
    :param extension_registry: Optional extension registry. If not given, the global extension registry will be used.
    :param data_store_params: Data store specific parameters.
    :return: A new data store instance
    """
    data_store_class = get_data_store_class(data_store_id, extension_registry=extension_registry)
    data_store_params_schema = data_store_class.get_data_store_params_schema()
    data_store_params = data_store_params_schema.from_instance(data_store_params) \
        if data_store_params else {}
    return data_store_class(**data_store_params)


def get_data_store_class(data_store_id: str,
                         extension_registry: Optional[ExtensionRegistry] = None) -> Type['DataStore']:
    """
    Get the class for the data store identified by *data_store_id*.

    :param data_store_id: A data store identifier.
    :param extension_registry: Optional extension registry. If not given, the global extension registry will be used.
    :return: The class for the data store.
    """
    extension_registry = extension_registry or get_extension_registry()
    if not extension_registry.has_extension(EXTENSION_POINT_DATA_STORES, data_store_id):
        raise DataStoreError(f'Unknown data store "{data_store_id}"')
    return extension_registry.get_component(EXTENSION_POINT_DATA_STORES, data_store_id)


def get_data_store_params_schema(data_store_id: str,
                                 extension_registry: Optional[ExtensionRegistry] = None) -> JsonObjectSchema:
    """
    Get the JSON schema for instantiating a new data store identified by *data_store_id*.

    :param data_store_id: A data store identifier.
    :param extension_registry: Optional extension registry. If not given, the global extension registry will be used.
    :return: The JSON schema for the data store's parameters.
    """
    data_store_class = get_data_store_class(data_store_id, extension_registry=extension_registry)
    return data_store_class.get_data_store_params_schema()


def find_data_store_extensions(predicate: ExtensionPredicate = None,
                               extension_registry: Optional[ExtensionRegistry] = None) -> List[Extension]:
    """
    Find data store extensions using the optional filter function *predicate*.

    :param predicate: An optional filter function.
    :param extension_registry: Optional extension registry. If not given, the global extension registry will be used.
    :return: List of data store extensions.
    """
    extension_registry = extension_registry or get_extension_registry()
    return extension_registry.find_extensions(EXTENSION_POINT_DATA_STORES, predicate=predicate)


#######################################################
# Classes
#######################################################

class DataStore(DataOpener, ABC):
    """
    A data store represents a collection of data resources that can be enumerated, queried, and opened in order to
    obtain in-memory representations.

    A store implementation may use any existing openers/writers, or define its own,
    or not use any openers/writers at all.

    DataStore is an abstract base class that both read-only and mutable data stores must implement.
    """

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        """
        Get descriptions of parameters that must or can be used to instantiate a new DataStore object.
        Parameters are named and described by the properties of the returned JSON object schema.
        The default implementation returns JSON object schema that can have any properties.
        """
        return JsonObjectSchema()

    @classmethod
    @abstractmethod
    def get_type_ids(cls) -> Tuple[str, ...]:
        """
        Get a tuple of supported data type identifiers.
        The first entry in the tuple represents this store's default data type.

        :return: The tuple of supported data type identifiers.
        """

    @abstractmethod
    def get_data_ids(self, type_id: str = None) -> Iterator[Tuple[str, Optional[str]]]:
        """
        Get an iterator over the data resource identifiers for the given type *type_id*.
        If *type_id* is omitted, all data resource identifiers are returned.

        If a store implementation supports only a single data type, it should verify that *type_id* is either None
        or equal to that single data type.

        The returned iterator items are 2-tuples of the form (*data_id*, *title*), where *data_id*
        is the actual data identifier and *title* is an optional, human-readable title for the data.

        :return: An iterator over the identifiers and titles of data resources provided by this data store.
        :raise DataStoreError: If an error occurs.
        """

    @abstractmethod
    def has_data(self, data_id: str) -> bool:
        """
        Check if the data resource given by *data_id* is available in this store.
        :return: True, if the data resource is available in this store, False otherwise.
        """

    @abstractmethod
    def describe_data(self, data_id: str) -> DataDescriptor:
        """
        Get the descriptor for the data resource given by *data_id*.

        Raises if *data_id* does not exist in this store.

        :return a data-type specific data descriptor
        :raise DataStoreError: If an error occurs.
        """

    @classmethod
    def get_search_params_schema(cls) -> JsonObjectSchema:
        """
        Get the schema for the parameters that can be passed as *search_params* to :meth:search_data().
        Parameters are named and described by the properties of the returned JSON object schema.
        The default implementation returns JSON object schema that can have any properties.

        :return: A JSON object schema whose properties describe this store's search parameters.
        """
        return JsonObjectSchema()

    @abstractmethod
    def search_data(self, type_id: str = None, **search_params) -> Iterator[DataDescriptor]:
        """
        Search this store for data resources.
        If *type_id* is given, the search is restricted to data resources of that type.

        Returns an iterator over the search results.
        The returned data descriptors may contain less information than returned by the :meth:describe_data()
        method.

        If a store implementation supports only a single data type, it should verify that *type_id* is either None
        or equal to that single data type.

        :param type_id: An optional data type identifier that is known to be supported by this data store.
        :param search_params: The search parameters.
        :return: An iterator of data descriptors for the found data resources.
        :raise DataStoreError: If an error occurs.
        """

    @abstractmethod
    def get_data_opener_ids(self, data_id: str = None, type_id: str = None) -> Tuple[str, ...]:
        """
        Get identifiers of data openers that can be used to open data resources from this store.

        If *data_id* is given, data accessors are restricted to the ones that can open the identified data resource.
        Raises if *data_id* does not exist in this store.

        If *type_id* is given, only openers that support this data type are returned.

        If a store implementation supports only a single data type, it should verify that *type_id* is either None
        or equal to that single data type.

        :param data_id: An optional data resource identifier that is known to exist in this data store.
        :param type_id: An optional data type identifier that is known to be supported by this data store.
        :return: A tuple of identifiers of data openers that can be used to open data resources.
        :raise DataStoreError: If an error occurs.
        """

    @abstractmethod
    def get_open_data_params_schema(self, data_id: str = None, opener_id: str = None) -> JsonObjectSchema:
        """
        Get the schema for the parameters passed as *open_params* to :meth:open_data(data_id, open_params).

        If *data_id* is given, the returned schema will be tailored to the constraints implied by the
        identified data resource. Some openers might not support this, therefore *data_id* is optional, and if
        it is omitted, the returned schema will be less restrictive. If given, the method raises
        if *data_id* does not exist in this store.

        If *opener_id* is given, the returned schema will be tailored to the constraints implied by the
        identified opener. Some openers might not support this, therefore *opener_id* is optional, and if
        it is omitted, the returned schema will be less restrictive.

        :param data_id: An optional data identifier that is known to exist in this data store.
        :param opener_id: An optional data opener identifier.
        :return: The schema for the parameters in *open_params*.
        :raise DataStoreError: If an error occurs.
        """

    @abstractmethod
    def open_data(self,
                  data_id: str,
                  opener_id: str = None,
                  **open_params) -> Any:
        """
        Open the data given by the data resource identifier *data_id* using the supplied *open_params*.

        The data type of the return value depends on the data opener used to open the data resource.

        If *opener_id* is given, the identified data opener will be used to open the data resource and
        *open_params* must comply with the schema of the opener's parameters. Note that some store
        implementations may not support using different openers or just support a single one.

        Raises if *data_id* does not exist in this store.

        :param data_id: The data identifier that is known to exist in this data store.
        :param opener_id: An optional data opener identifier.
        :param open_params: Opener-specific parameters.
        :return: An in-memory representation of the data resources identified by *data_id* and *open_params*.
        :raise DataStoreError: If an error occurs.
        """


class MutableDataStore(DataStore, DataWriter, ABC):
    """
    A mutable data store is a data store that also allows for adding, updating, and removing data resources.

    MutableDataStore is an abstract base class that any mutable data store must implement.
    """

    @abstractmethod
    def get_data_writer_ids(self, type_id: str = None) -> Tuple[str, ...]:
        """
        Get identifiers of data writers that can be used to write data resources to this store.

        If *type_id* is given, only writers that support this data type are returned.

        If a store implementation supports only a single data type, it should verify that *type_id* is either None
        or equal to that single data type.

        :param type_id: An optional data type identifier that is known to be supported by this data store.
        :return: A tuple of identifiers of data writers that can be used to write data resources.
        :raise DataStoreError: If an error occurs.
        """

    @abstractmethod
    def get_write_data_params_schema(self, writer_id: str = None) -> JsonObjectSchema:
        """
        Get the schema for the parameters passed as *write_params* to
        :meth:write_data(data, data_id, open_params).

        If *writer_id* is given, the returned schema will be tailored to the constraints implied by the
        identified writer. Some writers might not support this, therefore *writer_id* is optional, and if
        it is omitted, the returned schema will be less restrictive.

        Given here is a pseudo-code implementation for stores that support multiple writers
        and where the store has common parameters with the writer:

            store_params_schema = self.get_data_store_params_schema()
            writer_params_schema = get_writer(writer_id).get_write_data_params_schema()
            return subtract_param_schemas(writer_params_schema, store_params_schema)

        :param writer_id: An optional data writer identifier.
        :return: The schema for the parameters in *write_params*.
        :raise DataStoreError: If an error occurs.
        """

    @abstractmethod
    def write_data(self,
                   data: Any,
                   data_id: str = None,
                   writer_id: str = None,
                   replace: bool = False,
                   **write_params) -> str:
        """
        Write a data in-memory instance using the supplied *data_id* and *write_params*.

        If data identifier *data_id* is not given, a writer-specific default will be generated, used, and returned.

        If *writer_id* is given, the identified data writer will be used to write the data resource and
        *write_params* must comply with the schema of writers's parameters. Note that some store
        implementations may not support using different writers or just support a single one.

        Given here is a pseudo-code implementation for stores that support multiple writers:

            writer_id = writer_id or self.gen_data_id()
            path = self.resolve_data_id_to_path(data_id)
            write_params = add_params(self.get_data_store_params(), write_params)
            get_writer(writer_id).write_data(data, path, **write_params)
            self.register_data(data_id, data)

        Raises if *data_id* does not exist in this store.

        :param data: The data in-memory instance to be written.
        :param data_id: An optional data identifier that is known to be unique in this data store.
        :param writer_id: An optional data writer identifier.
        :param replace: Whether to replace an existing data resource.
        :param write_params: Writer-specific parameters.
        :return: The data identifier used to write the data.
        :raise DataStoreError: If an error occurs.
        """

    @abstractmethod
    def delete_data(self, data_id: str):
        """
        Delete the data resource identified by *data_id*.

        Typically, an implementation would delete the data resource from the physical storage
        and also remove any registered metadata from an associated database.

        Raises if *data_id* does not exist in this store.

        :param data_id: An data identifier that is known to exist in this data store.
        :raise DataStoreError: If an error occurs.
        """

    @abstractmethod
    def register_data(self, data_id: str, data: Any):
        """
        Register the in-memory representation of a data resource *data* using the given
        data resource identifier *data_id*.

        This method can be used to register data resources that are already physically
        stored in the data store, but are not yet searchable or otherwise accessible by
        the given *data_id*.

        Typically, an implementation would extract metadata from *data* and store it in a
        store-specific database. An implementation should just store the metadata of *data*.
        It should not write *data*.

        :param data_id: A data resource identifier that is known to be unique in this data store.
        :param data: An in-memory representation of a data resource.
        :raise DataStoreError: If an error occurs.
        """

    @abstractmethod
    def deregister_data(self, data_id: str):
        """
        De-register a data resource identified by *data_id* from this data store.

        This method can be used to de-register data resources so it will be no longer
        searchable or otherwise accessible by the given *data_id*.

        Typically, an implementation would extract metadata from *data* and store it in a
        store-specific database. An implementation should only remove a data resource's metadata.
        It should not delete *data* from its physical storage space.

        Raises if *data_id* does not exist in this store.

        :param data_id: A data resource identifier that is known to exist in this data store.
        :raise DataStoreError: If an error occurs.
        """
