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
from typing import Dict, Any, Tuple, Optional

import s3fs
import xarray as xr

from xcube.core.store.accessor import DataAccessorError
from xcube.core.store.accessor import DataOpener
from xcube.core.store.accessor import DataWriter
from xcube.core.store.accessors.posix import PosixDataDeleterMixin
from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema


class DatasetNetcdfPosixDataAccessor(PosixDataDeleterMixin, DataWriter, DataOpener):
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

    def write_data(self, data: xr.Dataset, data_id: str, replace=False, **write_params):
        assert_instance(data, xr.Dataset, 'data')
        data.to_netcdf(data_id, **write_params)


class ZarrOpenerParamsSchemaMixin:

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                group=JsonStringSchema(
                    description='Group path. (a.k.a. path in zarr terminology.).',
                    min_length=1,
                ),
                chunks=JsonObjectSchema(
                    description='Optional chunk sizes along each dimension. Chunk size values may '
                                'be None, "auto" or an integer value.',
                    examples=[{'time': None, 'lat': 'auto', 'lon': 90},
                              {'time': 1, 'y': 512, 'x': 512}],
                    additional_properties=True,
                ),
                decode_cf=JsonBooleanSchema(
                    description='Whether to decode these variables, assuming they were saved '
                                'according to CF conventions.',
                    default=True,
                ),
                mask_and_scale=JsonBooleanSchema(
                    description='If True, replace array values equal to attribute "_FillValue" with NaN. '
                                'Use "scaling_factor" and "add_offset" attributes to compute actual values.',
                    default=True,
                ),
                decode_times=JsonBooleanSchema(
                    description='If True, decode times encoded in the standard NetCDF datetime format '
                                'into datetime objects. Otherwise, leave them encoded as numbers.',
                    default=True,
                ),
                decode_coords=JsonBooleanSchema(
                    description='If True, decode the \"coordinates\" attribute to identify coordinates in '
                                'the resulting dataset.',
                    default=True,
                ),
                drop_variables=JsonArraySchema(
                    items=JsonStringSchema(min_length=1),
                ),
                consolidated=JsonBooleanSchema(
                    description='Whether to open the store using zarr\'s consolidated metadata '
                                'capability. Only works for stores that have already been consolidated.',
                    default=False,
                ),
            ),
            required=[],
            additional_properties=False
        )


class ZarrWriterParamsSchemaMixin:

    # noinspection PyMethodMayBeStatic
    def get_write_data_params_schema(self) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                group=JsonStringSchema(
                    description='Group path. (a.k.a. path in zarr terminology.).',
                    min_length=1,
                ),
                encoding=JsonObjectSchema(
                    description='Nested dictionary with variable names as keys and '
                                'dictionaries of variable specific encodings as values.',
                    examples=[{'my_variable': {'dtype': 'int16', 'scale_factor': 0.1, }}],
                    additional_properties=True,
                ),
                consolidated=JsonBooleanSchema(
                    description='If True, apply zarr’s consolidate_metadata() '
                                'function to the store after writing.'
                ),
                append_dim=JsonStringSchema(
                    description='If set, the dimension on which the data will be appended.',
                    min_length=1,
                )
            ),
            required=[],
            additional_properties=False
        )


class DatasetZarrPosixAccessor(ZarrOpenerParamsSchemaMixin,
                               ZarrWriterParamsSchemaMixin,
                               PosixDataDeleterMixin,
                               DataWriter,
                               DataOpener):
    """
    Extension name: "dataset:zarr:posix"
    """

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        return xr.open_zarr(data_id, **open_params)

    def write_data(self, data: xr.Dataset, data_id: str, replace=False, **write_params):
        assert_instance(data, xr.Dataset, 'data')
        data.to_zarr(data_id, **write_params)


#######################################################
# xr.Dataset / Zarr S3
#######################################################

class S3Mixin:
    """
    Provides common S3 parameters.
    """

    @classmethod
    def get_s3_params_schema(self) -> JsonObjectSchema:
        # TODO: Use defaults as described in
        #   https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html
        return JsonObjectSchema(
            properties=dict(
                anon=JsonBooleanSchema(default=False),
                aws_access_key_id=JsonStringSchema(min_length=1),
                aws_secret_access_key=JsonStringSchema(min_length=1),
                aws_session_token=JsonStringSchema(min_length=1),
                bucket_name=JsonStringSchema(min_length=1),
                region_name=JsonStringSchema(min_length=1),
            ),
        )

    @classmethod
    def consume_s3fs_params(cls, params: Dict[str, Any]) -> Tuple[s3fs.S3FileSystem, Dict[str, Any]]:
        aws_access_key_id = params.pop('aws_access_key_id', None)
        aws_secret_access_key = params.pop('aws_secret_access_key', None)
        aws_session_token = params.pop('aws_session_token', None)
        anon = params.pop('anon', not any((aws_access_key_id, aws_secret_access_key, aws_session_token)))
        client_kwargs = dict(region_name=params.pop('region_name', None))
        return s3fs.S3FileSystem(anon=anon,
                                 key=aws_access_key_id,
                                 secret=aws_secret_access_key,
                                 token=aws_session_token,
                                 client_kwargs=client_kwargs), params

    @classmethod
    def consume_bucket_name_param(cls, params: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
        bucket_name = params.pop('bucket_name', None)
        return bucket_name, params


class DatasetZarrS3Accessor(ZarrOpenerParamsSchemaMixin,
                            ZarrWriterParamsSchemaMixin,
                            S3Mixin,
                            DataWriter,
                            DataOpener):
    """
    Opener and Writer extension with name "dataset:zarr:s3".

    :param s3_fs: Optional, pre-computed instance of ``s3fs.S3FileSystem``.
    """

    def __init__(self, s3_fs: s3fs.S3FileSystem = None):
        self._s3_fs = s3_fs

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        schema = super().get_open_data_params_schema(data_id)
        if self._s3_fs is None:
            # If there is no S3 FS yet, we need extra S3 parameters to create it
            schema.properties.update(self.get_s3_params_schema())
        else:
            # Note: here we might have a look at given data_id and return data-specific open params.
            pass
        return schema

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        s3_fs = self._s3_fs
        if s3_fs is None:
            s3_fs, open_params = self.consume_s3fs_params(open_params)
        bucket_name, open_params = self.consume_bucket_name_param(open_params)
        try:
            return xr.open_zarr(s3fs.S3Map(root=f'{bucket_name}/{data_id}' if bucket_name else data_id,
                                           s3=s3_fs,
                                           check=False),
                                **open_params)
        except ValueError as e:
            raise DataAccessorError(f'{e}') from e

    # noinspection PyMethodMayBeStatic
    def get_write_data_params_schema(self) -> JsonObjectSchema:
        schema = super().get_write_data_params_schema()
        if self._s3_fs is None:
            # If there is no S3 FS yet, we need extra S3 parameters to create it
            schema.properties.update(self.get_s3_params_schema())
        return schema

    def write_data(self, data: xr.Dataset, data_id: str, replace=False, **write_params):
        assert_instance(data, xr.Dataset, 'data')
        s3_fs = self._s3_fs
        if s3_fs is None:
            s3_fs, write_params = self.consume_s3fs_params(write_params)
        bucket_name, write_params = self.consume_bucket_name_param(write_params)
        try:
            data.to_zarr(s3fs.S3Map(root=f'{bucket_name}/{data_id}' if bucket_name else data_id,
                                    s3=s3_fs,
                                    check=False),
                         mode='w' if replace else None,
                         **write_params)
        except ValueError as e:
            raise DataAccessorError(f'{e}') from e

    def delete_data(self, data_id: str):
        # TODO: implement me
        raise NotImplementedError()
