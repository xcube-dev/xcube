# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from abc import abstractmethod, ABC

import xarray as xr

from xcube.util.jsonschema import JsonObjectSchema

# Name of the (instance) method that processes a
# dataset and returns a new one.
# The one and only argument of the method is an xarray.Dataset.
# Then any number of processing parameters can be passed as
# keyword-arguments.
# See :meth:`DatasetProcessor#process_dataset`.
METHOD_NAME_DATASET_PROCESSOR = "process_dataset"

# Name of the (static/class) method that returns
# the JSON Object Schema for the processor parameters.
# The methods takes no arguments.
# See :meth:`DatasetProcessor#get_process_params_schema`.
METHOD_NAME_PARAMS_SCHEMA_GETTER = "get_process_params_schema"


class DatasetProcessor(ABC):
    """A generic dataset processor."""

    @abstractmethod
    def process_dataset(self, dataset: xr.Dataset, **process_params) -> xr.Dataset:
        """Process *dataset* into a new dataset using the
        given processing parameters *process_params*.

        The passed *process_params* must validate against
        the JSON schema returned by the :meth:`get_process_params_schema`
        class method.

        Args:
            dataset: The dataset to be processed.
            **process_params: The processing parameters.

        Returns:
            A new dataset.
        """

    @classmethod
    def get_process_params_schema(cls) -> JsonObjectSchema:
        """Get the JSON Schema for the processing parameters passed
        to :meth:`process_dataset`.

        The parameters are described by the JSON Object Schema's
        properties.

        The default implementation returns a JSON Schema
        that allows for any parameters of any type.

        Returns:
            a JSON Object Schema.
        """
        return JsonObjectSchema(additional_properties=True)
