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
from typing import Dict, Any

import xarray as xr

from xcube.core.store.accessor import DataOpener
from xcube.core.store.accessor import DataWriter
from xcube.core.store.accessors.posix import PosixDataDeleter
from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonObjectSchema


class DatasetNetcdfPosixDataAccessor(PosixDataDeleter, DataWriter, DataOpener):
    """
    Extension name: "dataset:netcdf:posix"
    """

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        # TODO: implement me
        return JsonObjectSchema()

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        return xr.open_dataset(data_id, **open_params)

    def get_write_data_params_schema(self) -> JsonObjectSchema:
        # TODO: implement me
        return JsonObjectSchema()

    def write_data(self, data: xr.Dataset, data_id: str, **write_params):
        assert_instance(data, xr.Dataset, 'data')
        data.to_netcdf(data_id, **write_params)


class DatasetZarrPosixAccessor(PosixDataDeleter, DataWriter, DataOpener):
    """
    Extension name: "dataset:zarr:posix"
    """

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        # TODO: implement me
        return JsonObjectSchema()

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        return xr.open_zarr(data_id, **open_params)

    def get_write_data_params_schema(self) -> JsonObjectSchema:
        # TODO: implement me
        return JsonObjectSchema()

    def write_data(self, data: xr.Dataset, data_id: str, **write_params):
        assert_instance(data, xr.Dataset, 'data')
        data.to_zarr(data_id, **write_params)


#######################################################
# xr.Dataset / Zarr S3
#######################################################

class DatasetZarrS3Accessor(DataWriter, DataOpener):
    """
    Extension name: "dataset:zarr:s3"
    """

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        # TODO: implement me
        return JsonObjectSchema()

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        import s3fs
        s3, open_params = _get_s3fs_and_consume_params(open_params)
        return xr.open_zarr(s3fs.S3Map(root=data_id, s3=s3, check=False), **open_params)

    def get_write_data_params_schema(self) -> JsonObjectSchema:
        # TODO: implement me
        return JsonObjectSchema()

    def write_data(self, data: xr.Dataset, data_id: str, **write_params):
        assert_instance(data, xr.Dataset, 'data')
        import s3fs
        s3, write_params = _get_s3fs_and_consume_params(write_params)
        data.to_zarr(s3fs.S3Map(root=data_id, s3=s3, check=False), **write_params)

    def delete_data(self, data_id: str):
        # TODO: implement me
        raise NotImplementedError()


def _get_s3fs_and_consume_params(params: Dict[str, Any]):
    import s3fs
    key = params.pop('key', params.pop('aws_access_key_id', None))
    secret = params.pop('secret', params.pop('aws_secret_access_key', None))
    token = params.pop('token', params.pop('aws_access_key_token', None))
    anon = params.pop('anon', key is None and secret is None and token is None)
    client_kwargs = dict(region_name=params.pop('region_name', None))
    return s3fs.S3FileSystem(anon=anon, key=key, secret=secret, token=token, client_kwargs=client_kwargs), params
