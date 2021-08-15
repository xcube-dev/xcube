# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
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

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

import fsspec

from xcube.core.store import DataOpener
from xcube.core.store import DataStoreError
from xcube.core.store import DataWriter
from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonIntegerSchema
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema

COMMON_FS_PARAMS_SCHEMA_PROPERTIES = dict(
    # passed to ``DirCache``, if the implementation supports
    # directory listing caching. Pass use_listings_cache=False
    # to disable such caching.
    use_listings_cache=JsonBooleanSchema(),
    listings_expiry_time=JsonNumberSchema(),
    max_paths=JsonIntegerSchema(),
    # If this is a cachable implementation, pass True here to force
    # creating a new instance even if a matching instance exists, and prevent
    # storing this instance.
    skip_instance_cache=JsonBooleanSchema(),
    asynchronous=JsonBooleanSchema(),
)

FS_PROTOCOL_PARAM_NAME = 'fs_protocol'
FS_PARAMS_PARAM_NAME = 'fs_params'
FS_PARAM_NAME = 'fs'


class FsAccessor:
    """
    Base class for accessing some file system.
    """

    @classmethod
    def get_fs_protocol(cls) -> str:
        """Get the storage identifier (file system protocol)."""
        return 'abstract'

    @classmethod
    def get_fs_params_schema(cls) -> JsonObjectSchema:
        """Get the JSON schema of the file system parameters."""
        return JsonObjectSchema(
            properties=COMMON_FS_PARAMS_SCHEMA_PROPERTIES,
            additional_properties=True,
        )

    @classmethod
    def get_fs(cls, params: Dict[str, Any]) \
            -> Tuple[fsspec.AbstractFileSystem, Dict[str, Any]]:

        fs = params.pop(FS_PARAM_NAME, None)
        if fs is not None:
            assert_instance(fs, fsspec.AbstractFileSystem, name=FS_PARAM_NAME)
            return fs

        fs_protocol = cls.get_fs_protocol()
        if fs_protocol == 'abstract':
            fs_protocol = params.pop(FS_PROTOCOL_PARAM_NAME, None)
            if fs_protocol is None:
                raise DataStoreError(f"cannot determine file system,"
                                     f" try using parameter"
                                     f" {FS_PROTOCOL_PARAM_NAME!r}")

        fs_params = params.pop(FS_PARAMS_PARAM_NAME, None)
        if fs_params is not None:
            assert_instance(fs_params, dict, name=FS_PARAMS_PARAM_NAME)
        try:
            return fsspec.filesystem(fs_protocol,
                                     **(fs_params or {})), params
        except (ValueError, ImportError):
            raise DataStoreError(f"cannot instantiate"
                                 f" file system {fs_protocol!r}")


class FileFsAccessor(FsAccessor):

    @classmethod
    def get_fs_protocol(cls) -> str:
        return 'file'

    @classmethod
    def get_fs_params_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            # TODO: add file fs params
            properties=COMMON_FS_PARAMS_SCHEMA_PROPERTIES,
            additional_properties=True,
        )


class MemoryFsAccessor(FsAccessor):

    @classmethod
    def get_fs_protocol(cls) -> str:
        return 'memory'

    @classmethod
    def get_fs_params_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            # TODO: add memory fs params
            properties=COMMON_FS_PARAMS_SCHEMA_PROPERTIES,
            additional_properties=True,
        )


class S3FsAccessor(FsAccessor):
    # Note, this is for AWS only
    _regions = [
        ['Europe (Frankfurt)', 'eu-central-1'],
        ['Europe (Ireland)', 'eu-west-1'],
        ['Europe (London)', 'eu-west-2'],
        ['Europe (Milan)', 'eu-south-1'],
        ['Europe (Paris)', 'eu-west-3'],
        ['Europe (Stockholm)', 'eu-north-1'],
        ['Canada (Central)', 'ca-central-1'],
        ['Africa (Cape Town)', 'af-south-1'],
        ['US East (Ohio)', 'us-east-2'],
        ['US East (N. Virginia)', 'us-east-1'],
        ['US West (N. California)', 'us-west-1'],
        ['US West (Oregon)', 'us-west-2'],
        ['South America (São Paulo)', 'sa-east-1'],
        ['Asia Pacific (Hong Kong)', 'ap-east-1'],
        ['Asia Pacific (Mumbai)', 'ap-south-1'],
        ['Asia Pacific (Osaka-Local)', 'ap-northeast-3'],
        ['Asia Pacific (Seoul)', 'ap-northeast-2'],
        ['Asia Pacific (Singapore)', 'ap-southeast-1'],
        ['Asia Pacific (Sydney)', 'ap-southeast-2'],
        ['Asia Pacific (Tokyo)', 'ap-northeast-1'],
        ['Middle East (Bahrain)', 'me-south-1'],
        ['China (Beijing)', 'cn-north-1'],
        ['China (Ningxia)', 'cn-northwest-1'],
    ]

    @classmethod
    def get_fs_protocol(cls) -> str:
        return 's3'

    @classmethod
    def get_fs_params_schema(cls) -> JsonObjectSchema:
        # We may use here AWS S3 defaults as described in
        #   https://boto3.amazonaws.com/v1/documentation/api/
        #   latest/guide/configuration.html
        return JsonObjectSchema(
            properties=dict(
                anon=JsonBooleanSchema(
                    title='Whether to anonymously connect to AWS S3'
                ),
                key=JsonStringSchema(
                    min_length=1,
                    title='AWS access key identifier',
                    description='Can also be set in profile section'
                                ' of ~/.aws/config, or by environment'
                                ' variable AWS_ACCESS_KEY_ID'
                ),
                secret=JsonStringSchema(
                    min_length=1,
                    title='AWS secret access key',
                    description='Can also be set in profile section'
                                ' of ~/.aws/config, or by environment'
                                ' variable AWS_SECRET_ACCESS_KEY'
                ),
                token=JsonStringSchema(
                    min_length=1,
                    title='Session token.',
                    description='Can also be set in profile section'
                                ' of ~/.aws/config, or by environment'
                                ' variable AWS_SESSION_TOKEN'
                ),
                client_kwargs=JsonObjectSchema(
                    properties=dict(
                        endpoint_url=JsonStringSchema(
                            min_length=1,
                            format='uri',
                            title='Alternative endpoint URL'
                        ),
                        bucket_name=JsonStringSchema(
                            min_length=1,
                            title='Name of the bucket'
                        ),
                        profile_name=JsonStringSchema(
                            min_length=1,
                            title='Name of the AWS configuration profile',
                            description='Section name with within'
                                        ' ~/.aws/config file,'
                                        ' which provides AWS configurations'
                                        ' and credentials.'
                        ),
                        region_name=JsonStringSchema(
                            min_length=1,
                            default='eu-central-1',
                            enum=[r[1] for r in cls._regions],
                            title='AWS storage region name'
                        ),
                    ),
                    additional_properties=True,
                ),
                **COMMON_FS_PARAMS_SCHEMA_PROPERTIES,
            ),
            additional_properties=True,
        )


class FsDataAccessor(DataOpener,
                     DataWriter,
                     FsAccessor,
                     ABC):
    """
    Abstract base class for data accessors that
    use an underlying file system.
    """

    @classmethod
    @abstractmethod
    def get_type_specifier(cls) -> str:
        """
        Get the data type specifier,
        for example "dataset" or "geodataframe".
        """

    @classmethod
    @abstractmethod
    def get_format_id(cls) -> str:
        """
        Get the format identifier,
        for example "zarr" or "geojson".
        """

    def get_delete_data_params_schema(self, data_id: str = None) \
            -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                recursive=JsonBooleanSchema(),
                maxdepth=JsonIntegerSchema(),
                fs_params=self.get_fs_params_schema(),
            ),
            additional_properties=False,
        )

    def delete_data(self,
                    data_id: str,
                    **delete_params):
        fs, delete_params = self.get_fs(delete_params)
        # Note: the default implementation may not be appropriate
        # for all data types. For example recursive=True
        # - for Zarr directory
        # - for Shapefile parent(!) directory
        fs.delete(data_id, **delete_params)
