from typing import Iterator

from xcube.util.jsonschema import JsonObjectSchema
from .descriptor import DataDescriptor
from .error import DataStoreError


# noinspection PyUnresolvedReferences
class DefaultSearchMixin:
    """
    A mixin for data store implementations that implements default search behaviour.
    It is expected that such data stores have no dedicated search parameters.
    """

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def get_search_params_schema(self, type_specifier: str = None) -> JsonObjectSchema:
        """
        Get search parameters JSON object schema.

        The default implementation returns a schema that does not allow for any search parameters.

        :param type_specifier:
        :return: a JSON object schema for the search parameters
        """
        return JsonObjectSchema(additional_properties=False)

    def search_data(self, type_specifier: str = None, **search_params) -> Iterator[DataDescriptor]:
        """
        Search the data store.

        The default implementation returns all data resources that may be filtered using
        the optional *type_specifier*.

        :param type_specifier: Type specifier to filter returned data resources.
        :param search_params: Not supported (yet)
        :return: an iterator of :class:DataDescriptor instances
        """
        if search_params:
            raise DataStoreError(f'Unsupported search parameters: {", ".join(search_params.keys())}')
        for data_id in self.get_data_ids(type_specifier=type_specifier):
            yield self.describe_data(data_id)
