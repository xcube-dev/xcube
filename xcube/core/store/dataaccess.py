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
from abc import ABCMeta, abstractmethod
from typing import Any, Dict

import geopandas as gpd
import xarray as xr

from xcube.util.jsonschema import JsonObjectSchema


class DatasetOpener(metaclass=ABCMeta):

    @property
    def open_dataset_params_schema(self) -> JsonObjectSchema:
        # TODO
        return JsonObjectSchema()

    def open_dataset(self, path: str, **open_params) -> xr.Dataset:
        return xr.open_dataset(path, **open_params)


class AbstractDatasetWriter(metaclass=ABCMeta):

    @property
    def write_dataset_params_schema(self) -> JsonObjectSchema:
        return JsonObjectSchema()

    @abstractmethod
    def write_dataset(self, dataset: xr.Dataset, path: str, **write_params):
        raise NotImplementedError()


#######################################################
# xr.Dataset / Netcdf
#######################################################

class NetcdfDatasetWriter(AbstractDatasetWriter):
    @property
    def write_dataset_params_schema(self) -> JsonObjectSchema:
        # TODO
        return JsonObjectSchema()

    def write_dataset(self, dataset: xr.Dataset, path: str, **write_params):
        dataset.to_netcdf(path, **write_params)


#######################################################
# xr.Dataset / Zarr
#######################################################

class ZarrDatasetOpener(DatasetOpener):
    @property
    def write_dataset_params_schema(self) -> JsonObjectSchema:
        # TODO
        return JsonObjectSchema()

    def open_dataset(self, path: str, **open_params) -> xr.Dataset:
        return xr.open_zarr(path, **open_params)


class ZarrDatasetWriter(AbstractDatasetWriter):
    @property
    def write_dataset_params_schema(self) -> JsonObjectSchema:
        # TODO
        return JsonObjectSchema()

    def write_dataset(self, dataset: xr.Dataset, path: str, **write_params):
        dataset.to_zarr(path, **write_params)


#######################################################
# xr.Dataset / Zarr N5
#######################################################

class ZarrN5DatasetOpener(DatasetOpener):

    def open_dataset(self, path: str, **open_params) -> xr.Dataset:
        normalize_keys = open_params.pop('normalize_keys', False)
        from zarr.n5 import N5Store
        return xr.open_zarr(N5Store(path, normalize_keys=normalize_keys), **open_params)


class ZarrN5DatasetWriter(AbstractDatasetWriter):
    @property
    def write_dataset_params_schema(self) -> JsonObjectSchema:
        # TODO
        return JsonObjectSchema()

    def write_dataset(self, dataset: xr.Dataset, path: str, **write_params):
        normalize_keys = write_params.pop('normalize_keys', False)
        from zarr.n5 import N5Store
        dataset.to_zarr(N5Store(path, normalize_keys=normalize_keys), **write_params)


#######################################################
# xr.Dataset / Zarr S3
#######################################################

class ZarrS3DatasetOpener(ZarrDatasetOpener):

    def open_dataset(self, path: str, **open_params) -> xr.Dataset:
        import s3fs
        s3, open_params = _get_s3_and_consume_params(open_params)
        return xr.open_zarr(s3fs.S3Map(root=path, s3=s3, check=False), **open_params)


class ZarrS3DatasetWriter(ZarrDatasetWriter):
    @property
    def write_dataset_params_schema(self) -> JsonObjectSchema:
        # TODO
        return None

    def write_dataset(self, dataset: xr.Dataset, path: str, **write_params):
        import s3fs
        s3, write_params = _get_s3_and_consume_params(write_params)
        dataset.to_zarr(s3fs.S3Map(root=path, s3=s3, check=False), **write_params)


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
    def open_geo_data_frame_params_schema(self) -> JsonObjectSchema:
        return JsonObjectSchema()

    def open_geo_data_frame(self, path: str, **open_params) -> gpd.GeoDataFrame:
        return gpd.read_file(path, **open_params)


class GeoDataFrameWriter:
    @property
    def write_geo_data_frame_params_schema(self) -> JsonObjectSchema:
        return JsonObjectSchema()

    def write_geo_data_frame(self, geo_data_frame: gpd.GeoDataFrame, path: str, **write_params):
        return geo_data_frame.to_file(path, **write_params)


#######################################################
# gpd.GeoDataFrame / GeoJSON
#######################################################

class GeoJsonGeoDataFrameWriter(GeoDataFrameWriter):

    def write_geo_data_frame(self, geo_data_frame: gpd.GeoDataFrame, path: str, **write_params):
        return geo_data_frame.to_file(path, driver='GeoJSON', **write_params)
