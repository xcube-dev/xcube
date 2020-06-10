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
from typing import Sequence

import xarray as xr

from xcube.core.store.dataaccess import DatasetOpener
from xcube.core.store.dataaccess import DatasetWriter
from xcube.util.jsonschema import JsonObjectSchema


class DataStore(ABC):
    """
    An abstract data store.
    """

    @classmethod
    def get_params_schema(cls) -> JsonObjectSchema:
        """
        Get descriptions of parameters that must or can be used to instantiate a new data store object.
        Parameters are named and described by the properties of the returned JSON object schema.
        The default implementation returns JSON object schema that can have any properties.
        """
        return JsonObjectSchema()


class DatasetStore(DataStore, DatasetOpener, ABC):

    @abstractmethod
    def get_dataset_opener_ids(self, dataset_id: str = None) -> Sequence[str]:
        """
        Get identifiers of data accessors that can be used to open datasets.
        If *dataset_id* is not given, all data accessors that can open datasets are returned.
        If *dataset_id* is given, data accessors are restricted to the ones that can open the identified dataset.

        :param dataset_id: An optional dataset identifier.
        :return: A sequence of identifiers of dataset openers that can be used to open datasets.
        """

    @abstractmethod
    def get_open_dataset_params_schema(self, dataset_id: str = None, opener_id: str = None) -> JsonObjectSchema:
        """
        Get the schema for the parameters passed as *open_params* to :meth:open_dataset(dataset_id, open_params).
        If *dataset_id* is given, the returned schema will be tailored to the constraints implied by the
        identified dataset. Some openers might not support this, therefore *dataset_id* is optional, and if
        it is omitted, the returned schema will be less restrictive.

        :param dataset_id: An optional dataset identifier.
        :param opener_id: An optional dataset opener identifier.
        :return: The schema for the parameters in *open_params*.
        """

    @abstractmethod
    def open_dataset(self, dataset_id: str, opener_id: str = None, **open_params) -> xr.Dataset:
        """
        Open the dataset given by the dataset identifier *dataset_id* using the supplied *open_params*.

        :param dataset_id: The dataset identifier.
        :param opener_id: An optional dataset opener identifier.
        :param open_params: Opener-specific parameters.
        :return: An xarray.Dataset instance.
        """


class MutableDatasetStore(DatasetStore, DatasetWriter, ABC):

    @abstractmethod
    def get_dataset_writer_ids(self) -> Sequence[str]:
        """
        Get identifiers of dataset writers.

        :return: A sequence of identifiers of dataset writers that can be used to write datasets.
        """

    @abstractmethod
    def get_write_dataset_params_schema(self, writer_id: str = None) -> JsonObjectSchema:
        """
        Get the schema for the parameters passed as *write_params* to
        :meth:write_dataset(dataset, dataset_id, open_params).

        :param writer_id: An optional dataset writer identifier.
        :return: The schema for the parameters in *write_params*.
        """

    @abstractmethod
    def write_dataset(self, dataset: xr.Dataset, dataset_id: str = None, writer_id: str = None, **write_params) -> str:
        """
        Write a dataset using the supplied *dataset_id* and *write_params*. If dataset identifier
        *dataset_id* is not given, a writer-specific default will be generated, used, and returned.

        :param dataset: The dataset instance to be written.
        :param dataset_id: An optional dataset identifier.
        :param writer_id: An optional dataset writer identifier.
        :param write_params: Writer-specific parameters.
        :return: The dataset identifier used to write the dataset.
        """
