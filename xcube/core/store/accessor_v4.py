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

import os.path
from abc import abstractmethod, ABC
from typing import Any, Dict, Optional, List

import geopandas as gpd
import pandas as pd
import xarray as xr

from xcube.constants import EXTENSION_POINT_DATA_OPENERS
from xcube.constants import EXTENSION_POINT_DATA_WRITERS
from xcube.core.dsio import rimraf
from xcube.util.assertions import assert_instance, assert_given
from xcube.util.extension import Extension
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.plugin import get_extension_registry


def get_data_opener(opener_id: str) -> 'DataOpener':
    """
    Get an instance of the data opener identified by *opener_id*.

    :param opener_id: The data opener identifier.
    :return: A data opener instance.
    """
    assert_given(opener_id, 'opener_id')
    return get_extension_registry().get_component(EXTENSION_POINT_DATA_OPENERS, opener_id)


def get_data_writer(writer_id: str) -> 'DataWriter':
    """
    Get an instance of the data opener identified by *writer_id*.

    :param writer_id: The data writer identifier.
    :return: A data writer instance.
    """
    assert_given(writer_id, 'writer_id')
    return get_extension_registry().get_component(EXTENSION_POINT_DATA_WRITERS, writer_id)


def get_data_opener_extensions(type_id: str = None,
                               format_id: str = None,
                               storage_id: str = None) -> List[Extension]:
    """
    Get information about registered data openers.

    :param type_id: Optional data type identifier to be supported.
    :param format_id: Optional data format identifier to be supported.
    :param storage_id: Optional data storage identifier to be supported.
    :return: List of matching extensions.
    """
    predicate = _get_data_accessor_predicate(type_id, format_id, storage_id)
    return get_extension_registry().find_extensions(EXTENSION_POINT_DATA_OPENERS, predicate=predicate)


def get_data_writer_extensions(type_id: str = None,
                               format_id: str = None,
                               storage_id: str = None) -> List[Extension]:
    """
    Get information about registered data writers for data type *type_id*.

    :param type_id: Optional data type identifier to be supported.
    :param format_id: Optional data format identifier to be supported.
    :param storage_id: Optional data storage identifier to be supported.
    :return: Mapping of opener identifiers to opener metadata.
    """
    predicate = _get_data_accessor_predicate(type_id, format_id, storage_id)
    return get_extension_registry().find_extensions(EXTENSION_POINT_DATA_WRITERS, predicate=predicate)


def _get_data_accessor_predicate(type_id: Optional[str], format_id: Optional[str], storage_id: Optional[str]):
    if not any((type_id, format_id, storage_id)):
        return None

    def predicate(extension: Extension) -> bool:
        parts = extension.name.split(':', maxsplit=4)
        if len(parts) < 3:
            raise ValueError(f'illegal data opener/writer extension name "{extension.name}"')
        type_ok = parts[0] == type_id if type_id else True
        format_ok = parts[1] == format_id if format_id else True
        storage_ok = parts[2] == storage_id if storage_id else True
        return type_ok and format_ok and storage_ok

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


class DataWriter(ABC):
    @abstractmethod
    def get_write_data_params_schema(self) -> JsonObjectSchema:
        """
        Get the schema for the parameters passed as *write_params* to
        :meth:write_data(data resource, data_id, open_params).

        :return: The schema for the parameters in *write_params*.
        """

    @abstractmethod
    def write_data(self, data: Any, data_id: str, replace: bool = False, **write_params):
        """
        Write a data resource using the supplied *data_id* and *write_params*.

        :param data: The data resource's in-memory representation to be written.
        :param data_id: A unique data resource identifier.
        :param replace: Whether to replace an existing data resource.
        :param write_params: Writer-specific parameters.
        :return: The data resource identifier used to write the data resource.
        """

    @abstractmethod
    def delete_data(self, data_id: str):
        """
        Delete a data resource.

        :param data_id: A data resource identifier known to exist.
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


#######################################################
# xr.Dataset:*:Posix
#######################################################

class GenericDatasetOpener(DataOpener):
    """
    Opener that opens data resources using the xarray.open_dataset(data_id, **open_params) function
    """

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        # TODO: implement me
        return JsonObjectSchema()

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        return xr.open_dataset(data_id, **open_params)


class PosixDataDeleterMixin:

    def delete_data(self, data_id: str):
        if not os.path.exists(data_id):
            raise FileNotFoundError(f'A dataset named "{data_id}" does not exist')
        rimraf(data_id)


#######################################################
# xr.Dataset:Netcdf:Posix
#######################################################

class NetcdfDatasetWriter(DataWriter, PosixDataDeleterMixin):
    def get_write_data_params_schema(self) -> JsonObjectSchema:
        # TODO: implement me
        return JsonObjectSchema()

    def write_data(self, data: xr.Dataset, data_id: str, **write_params):
        assert_instance(data, xr.Dataset, 'data')
        data.to_netcdf(data_id, **write_params)


#######################################################
# xr.Dataset:Zarr:Posix
#######################################################

class ZarrDatasetOpener(DataOpener):
    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        # TODO: implement me
        return JsonObjectSchema()

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        return xr.open_zarr(data_id, **open_params)


class ZarrDatasetWriter(DataWriter, PosixDataDeleterMixin):
    def get_write_data_params_schema(self) -> JsonObjectSchema:
        # TODO: implement me
        return JsonObjectSchema()

    def write_data(self, data: xr.Dataset, data_id: str, **write_params):
        assert_instance(data, xr.Dataset, 'data')
        data.to_zarr(data_id, **write_params)


#######################################################
# xr.Dataset:Zarr:N5
#######################################################

class ZarrN5DatasetOpener(ZarrDatasetOpener):

    def open_data(self, path: str, **open_params) -> xr.Dataset:
        normalize_keys = open_params.pop('normalize_keys', False)
        from zarr.n5 import N5Store
        return xr.open_zarr(N5Store(path, normalize_keys=normalize_keys), **open_params)


class ZarrN5DatasetWriter(ZarrDatasetWriter, PosixDataDeleterMixin):
    def get_write_data_params_schema(self) -> JsonObjectSchema:
        # TODO: implement me
        return JsonObjectSchema()

    def write_data(self, data: xr.Dataset, data_id: str, **write_params):
        assert_instance(data, xr.Dataset, 'data')
        normalize_keys = write_params.pop('normalize_keys', False)
        from zarr.n5 import N5Store
        data.to_zarr(N5Store(data_id, normalize_keys=normalize_keys), **write_params)


#######################################################
# xr.Dataset / Zarr S3
#######################################################

class ZarrS3DatasetOpener(ZarrDatasetOpener):
    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        # TODO: implement me
        return JsonObjectSchema()

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        import s3fs
        s3, open_params = _get_s3_and_consume_params(open_params)
        return xr.open_zarr(s3fs.S3Map(root=data_id, s3=s3, check=False), **open_params)


class ZarrS3DatasetWriter(ZarrDatasetWriter):
    def get_write_data_params_schema(self) -> JsonObjectSchema:
        # TODO: implement me
        return JsonObjectSchema()

    def write_data(self, data: xr.Dataset, data_id: str, **write_params):
        assert_instance(data, xr.Dataset, 'data')
        import s3fs
        s3, write_params = _get_s3_and_consume_params(write_params)
        data.to_zarr(s3fs.S3Map(root=data_id, s3=s3, check=False), **write_params)

    def delete_data(self, data_id: str):
        # TODO: implement me
        pass


def _get_s3_and_consume_params(params: Dict[str, Any]):
    import s3fs
    key = params.pop('key', params.pop('aws_access_key_id', None))
    secret = params.pop('secret', params.pop('aws_secret_access_key', None))
    token = params.pop('token', params.pop('aws_access_key_token', None))
    anon = params.pop('anon', key is None and secret is None and token is None)
    client_kwargs = dict(region_name=params.pop('region_name', None))
    return s3fs.S3FileSystem(anon=anon, key=key, secret=secret, token=token, client_kwargs=client_kwargs), params


#######################################################
# gpd.GeoDataFrame:*:Posix
#######################################################

class GeoDataFrameOpener(DataOpener):
    @property
    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        # TODO: implement me
        return JsonObjectSchema()

    def open_data(self, data_id: str, **open_params) -> gpd.GeoDataFrame:
        return gpd.read_file(data_id, **open_params)


class GeoDataFrameWriter(DataWriter, PosixDataDeleterMixin):
    def get_write_data_params_schema(self) -> JsonObjectSchema:
        # TODO: implement me
        return JsonObjectSchema()

    def write_data(self, data: gpd.GeoDataFrame, data_id: str, **write_params):
        assert_instance(data, (gpd.GeoDataFrame, pd.DataFrame), 'data')
        data.to_file(data_id, **write_params)


#######################################################
# gpd.GeoDataFrame:GeoJSON:Posix
#######################################################

class GeoJsonGeoDataFrameWriter(GeoDataFrameWriter):
    def get_write_data_params_schema(self) -> JsonObjectSchema:
        # TODO: implement me
        return JsonObjectSchema()

    def write_data(self, data: gpd.GeoDataFrame, gdf_id: str, **write_params):
        assert_instance(data, (gpd.GeoDataFrame, pd.DataFrame), 'data')
        data.to_file(gdf_id, driver='GeoJSON', **write_params)
