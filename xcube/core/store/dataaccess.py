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
from typing import Any, Dict

import geopandas as gpd
import xarray as xr

from xcube.core.store.descriptor import DatasetDescriptor
from xcube.util.jsonschema import JsonObjectSchema


#######################################################
# Interfaces
#######################################################

class DatasetDescriber(ABC):
    @abstractmethod
    def describe_dataset(self, dataset_id: str) -> DatasetDescriptor:
        """
        Descriptor for the dataset given by the dataset identifier *dataset_id*.

        :param dataset_id: The dataset identifier.
        :return: A dataset descriptor.
        """
        raise NotImplementedError()


class DatasetOpener(ABC):
    @abstractmethod
    def get_open_dataset_params_schema(self, dataset_id: str = None) -> JsonObjectSchema:
        """
        Get the schema for the parameters passed as *open_params* to :meth:open_dataset(dataset_id, open_params).
        If *dataset_id* is given, the returned schema will be tailored to the constraints implied by the
        identified dataset. Some openers might not support this, therefore *dataset_id* is optional, and if
        it is omitted, the returned schema will be less restrictive.

        :param dataset_id: An optional dataset identifier.
        :return: The schema for the parameters in *open_params*.
        """
        raise NotImplementedError()

    @abstractmethod
    def open_dataset(self, dataset_id: str, **open_params) -> xr.Dataset:
        """
        Open the dataset given by the dataset identifier *dataset_id* using the supplied *open_params*.

        :param dataset_id: The dataset identifier.
        :param open_params: Opener-specific parameters.
        :return: An xarray.Dataset instance.
        """
        raise NotImplementedError()


class DatasetWriter(ABC):
    @abstractmethod
    def get_write_dataset_params_schema(self) -> JsonObjectSchema:
        """
        Get the schema for the parameters passed as *write_params* to
        :meth:write_dataset(dataset, dataset_id, open_params).

        :return: The schema for the parameters in *write_params*.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_dataset(self, dataset: xr.Dataset, dataset_id: str = None, **write_params) -> str:
        """
        Write a dataset using the supplied *dataset_id* and *write_params*. If dataset identifier
        *dataset_id* is not given, a writer-specific default will be generated, used, and returned.

        :param dataset: The dataset instance to be written.
        :param dataset_id: An optional dataset identifier.
        :param write_params: Writer-specific parameters.
        :return: The dataset identifier used to write the dataset.
        """
        raise NotImplementedError()


#######################################################
# Base classes
#######################################################

class GenericDatasetOpener(DatasetOpener):
    def get_open_dataset_params_schema(self, dataset_id: str = None) -> JsonObjectSchema:
        # TODO
        return JsonObjectSchema()

    def open_dataset(self, dataset_id: str, **open_params) -> xr.Dataset:
        return xr.open_dataset(dataset_id, **open_params)


#######################################################
# xr.Dataset / Netcdf
#######################################################

class NetcdfDatasetWriter(DatasetWriter):
    def get_write_dataset_params_schema(self) -> JsonObjectSchema:
        # TODO
        return JsonObjectSchema()

    def write_dataset(self, dataset: xr.Dataset, dataset_id: str = None, **write_params) -> str:
        dataset_id = dataset_id or 'out.nc'
        dataset.to_netcdf(dataset_id, **write_params)
        return dataset_id


#######################################################
# xr.Dataset / Zarr
#######################################################

class ZarrDatasetOpener(DatasetOpener):
    def get_open_dataset_params_schema(self, dataset_id: str = None) -> JsonObjectSchema:
        # TODO
        return JsonObjectSchema()

    def open_dataset(self, dataset_id: str, **open_params) -> xr.Dataset:
        return xr.open_zarr(dataset_id, **open_params)


class ZarrDatasetWriter(DatasetWriter):
    def get_write_dataset_params_schema(self) -> JsonObjectSchema:
        # TODO
        return JsonObjectSchema()

    def write_dataset(self, dataset: xr.Dataset, dataset_id: str = None, **write_params) -> str:
        dataset_id = dataset_id or 'out.zarr'
        dataset.to_zarr(dataset_id, **write_params)
        return dataset_id


#######################################################
# xr.Dataset / Zarr N5
#######################################################

class ZarrN5DatasetOpener(ZarrDatasetOpener):

    def open_dataset(self, path: str, **open_params) -> xr.Dataset:
        normalize_keys = open_params.pop('normalize_keys', False)
        from zarr.n5 import N5Store
        return xr.open_zarr(N5Store(path, normalize_keys=normalize_keys), **open_params)


class ZarrN5DatasetWriter(ZarrDatasetWriter):
    def get_write_dataset_params_schema(self) -> JsonObjectSchema:
        # TODO
        return JsonObjectSchema()

    def write_dataset(self, dataset: xr.Dataset, dataset_id: str = None, **write_params):
        dataset_id = dataset_id or 'out.n5'
        normalize_keys = write_params.pop('normalize_keys', False)
        from zarr.n5 import N5Store
        dataset.to_zarr(N5Store(dataset_id, normalize_keys=normalize_keys), **write_params)
        return dataset_id


#######################################################
# xr.Dataset / Zarr S3
#######################################################

class ZarrS3DatasetOpener(ZarrDatasetOpener):
    def get_open_dataset_params_schema(self, dataset_id: str = None) -> JsonObjectSchema:
        # TODO
        return JsonObjectSchema()

    def open_dataset(self, dataset_id: str, **open_params) -> xr.Dataset:
        import s3fs
        s3, open_params = _get_s3_and_consume_params(open_params)
        return xr.open_zarr(s3fs.S3Map(root=dataset_id, s3=s3, check=False), **open_params)


class ZarrS3DatasetWriter(ZarrDatasetWriter):
    def get_write_dataset_params_schema(self) -> JsonObjectSchema:
        # TODO
        return JsonObjectSchema()

    def write_dataset(self, dataset: xr.Dataset, dataset_id: str = None, **write_params) -> str:
        dataset_id = dataset_id or 'out.zarr'
        import s3fs
        s3, write_params = _get_s3_and_consume_params(write_params)
        dataset.to_zarr(s3fs.S3Map(root=dataset_id, s3=s3, check=False), **write_params)
        return dataset_id


def _get_s3_and_consume_params(params: Dict[str, Any]):
    import s3fs
    key = params.pop('key', params.pop('aws_access_key_id', None))
    secret = params.pop('secret', params.pop('aws_secret_access_key', None))
    token = params.pop('token', params.pop('aws_access_key_token', None))
    anon = params.pop('anon', key is None and secret is None and token is None)
    client_kwargs = dict(region_name=params.pop('region_name', None))
    return s3fs.S3FileSystem(anon=anon, key=key, secret=secret, token=token, client_kwargs=client_kwargs), params


#######################################################
# gpd.GeoDataFrame
#######################################################

class GeoDataFrameOpener:
    @property
    def open_geo_data_frame_params_schema(self, gdf_id=None) -> JsonObjectSchema:
        # TODO
        return JsonObjectSchema()

    def open_geo_data_frame(self, gdf_id: str, **open_params) -> gpd.GeoDataFrame:
        return gpd.read_file(gdf_id, **open_params)


class GeoDataFrameWriter:
    def write_geo_data_frame_params_schema(self) -> JsonObjectSchema:
        # TODO
        return JsonObjectSchema()

    def write_geo_data_frame(self, geo_data_frame: gpd.GeoDataFrame, gdf_id: str = None, **write_params) -> str:
        gdf_id = gdf_id or 'out.shp'
        geo_data_frame.to_file(gdf_id, **write_params)
        return gdf_id


#######################################################
# gpd.GeoDataFrame / GeoJSON
#######################################################

class GeoJsonGeoDataFrameWriter(GeoDataFrameWriter):
    def write_geo_data_frame_params_schema(self) -> JsonObjectSchema:
        # TODO
        return JsonObjectSchema()

    def write_geo_data_frame(self, geo_data_frame: gpd.GeoDataFrame, gdf_id: str = None, **write_params) -> str:
        gdf_id = gdf_id or 'out.geojson'
        geo_data_frame.to_file(gdf_id, driver='GeoJSON', **write_params)
        return gdf_id
