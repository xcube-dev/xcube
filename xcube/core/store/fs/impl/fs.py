# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.util.jsonschema import (
    JsonBooleanSchema,
    JsonIntegerSchema,
    JsonObjectSchema,
    JsonStringSchema,
)

from ..accessor import COMMON_STORAGE_OPTIONS_SCHEMA_PROPERTIES, FsAccessor


class FileFsAccessor(FsAccessor):
    @classmethod
    def get_protocol(cls) -> str:
        return "file"

    @classmethod
    def get_storage_options_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                auto_mkdirs=JsonBooleanSchema(
                    description="Whether, when opening a file, the directory"
                    " containing it should be created (if it"
                    " doesn't already exist)."
                ),
                **COMMON_STORAGE_OPTIONS_SCHEMA_PROPERTIES,
            ),
            additional_properties=True,
        )


class MemoryFsAccessor(FsAccessor):
    @classmethod
    def get_protocol(cls) -> str:
        return "memory"


class S3FsAccessor(FsAccessor):
    @classmethod
    def get_protocol(cls) -> str:
        return "s3"

    @classmethod
    def get_storage_options_schema(cls) -> JsonObjectSchema:
        # We may use here AWS S3 defaults as described in
        #   https://boto3.amazonaws.com/v1/documentation/api/
        #   latest/guide/configuration.html
        return JsonObjectSchema(
            properties=dict(
                anon=JsonBooleanSchema(
                    title="Whether to anonymously connect to AWS S3."
                ),
                key=JsonStringSchema(
                    min_length=1,
                    title="AWS access key identifier.",
                    description="Can also be set in profile section"
                    " of ~/.aws/config, or by environment"
                    " variable AWS_ACCESS_KEY_ID.",
                ),
                secret=JsonStringSchema(
                    min_length=1,
                    title="AWS secret access key.",
                    description="Can also be set in profile section"
                    " of ~/.aws/config, or by environment"
                    " variable AWS_SECRET_ACCESS_KEY.",
                ),
                token=JsonStringSchema(
                    min_length=1,
                    title="Session token.",
                    description="Can also be set in profile section"
                    " of ~/.aws/config, or by environment"
                    " variable AWS_SESSION_TOKEN.",
                ),
                use_ssl=JsonBooleanSchema(
                    description="Whether to use SSL in connections to S3;"
                    " may be faster without, but insecure.",
                    default=True,
                ),
                requester_pays=JsonBooleanSchema(
                    description='If "RequesterPays" buckets are supported.',
                    default=False,
                ),
                s3_additional_kwargs=JsonObjectSchema(
                    description="parameters that are used when calling"
                    " S3 API methods. Typically used for"
                    ' things like "ServerSideEncryption".',
                    additional_properties=True,
                ),
                client_kwargs=JsonObjectSchema(
                    description="Parameters for the botocore client.",
                    properties=dict(
                        endpoint_url=JsonStringSchema(
                            min_length=1,
                            format="uri",
                            title="Alternative endpoint URL.",
                        ),
                        # bucket_name=JsonStringSchema(
                        #     min_length=1,
                        #     title='Name of the bucket'
                        # ),
                        profile_name=JsonStringSchema(
                            min_length=1,
                            title="Name of the AWS configuration profile",
                            description="Section name with within"
                            " ~/.aws/config file,"
                            " which provides AWS configurations"
                            " and credentials.",
                        ),
                        region_name=JsonStringSchema(
                            min_length=1, title="AWS storage region name"
                        ),
                    ),
                    additional_properties=True,
                ),
                **COMMON_STORAGE_OPTIONS_SCHEMA_PROPERTIES,
            ),
            additional_properties=True,
        )


class AzureFsAccessor(FsAccessor):
    @classmethod
    def get_protocol(cls) -> str:
        return "abfs"

    @classmethod
    def get_storage_options_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                anon=JsonBooleanSchema(
                    title="Whether to anonymously connect to Azure Blob Storage."
                ),
                account_name=JsonStringSchema(
                    min_length=1,
                    title="Azure storage account name.",
                    description="Must be used with the account key parameter."
                    " This is not required when using a"
                    " connection string.",
                ),
                account_key=JsonStringSchema(
                    min_length=1,
                    title="Azure storage account key.",
                    description="Must be used with the account"
                    " name parameter."
                    " This is not required when using a"
                    " connection string",
                ),
                connection_string=JsonStringSchema(
                    min_length=1,
                    title="Connection string for Azure blob storage.",
                    description="Use this parameter inplace of both"
                    " account name and key"
                    " because they are both contained"
                    " in the string.",
                ),
                **COMMON_STORAGE_OPTIONS_SCHEMA_PROPERTIES,
            ),
            additional_properties=True,
        )


class FtpFsAccessor(FsAccessor):
    @classmethod
    def get_protocol(cls) -> str:
        return "ftp"

    @classmethod
    def get_storage_options_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                host=JsonStringSchema(
                    title="FTP host",
                    description="The remote server name/ip to connect to",
                    min_length=1,
                ),
                port=JsonIntegerSchema(
                    minimum=0,
                    maximum=65535,
                    default=21,
                    title="FTP port",
                    description="Port to connect with",
                ),
                username=JsonStringSchema(
                    min_length=1,
                    title="User name",
                    description="If authenticating, the user's identifier",
                ),
                password=JsonStringSchema(
                    min_length=1,
                    title="User password",
                    description="User's password on the server, if using",
                ),
                **COMMON_STORAGE_OPTIONS_SCHEMA_PROPERTIES,
            ),
            additional_properties=True,
        )


class HttpsFsAccessor(FsAccessor):
    @classmethod
    def get_protocol(cls) -> str:
        return "https"
