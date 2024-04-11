# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import hashlib
from typing import Optional

from xcube.server.api import ApiHandler, ApiError
from .api import api
from .context import S3Context
from .listbucket import _LAST_MODIFIED_DUMMY
from .listbucket import dict_to_xml
from .listbucket import list_bucket_result_to_xml
from .listbucket import list_s3_bucket_v1
from .listbucket import list_s3_bucket_v2

BUCKET_PARAMETER = {
    "name": "bucket",
    "in": "path",
    "description": "The bucket name."
    ' The bucket "datasets" provides Zarr datasets,'
    ' the bucket "pyramids" provides them as'
    " multi-resolution image pyramids.",
    "schema": {
        "type": "string",
        "enum": ["datasets", "pyramids"],
    },
}

OPTIONAL_STRING_SCHEMA = {
    "type": ["string", "null"],
    "default": None,
}

LIST_BUCKET_PARAMETERS = [
    BUCKET_PARAMETER,
    {
        "name": "list-type",
        "in": "query",
        "description": "The operation version (1 or 2)",
        "schema": {
            "type": ["integer", "null"],
            "enum": [None, 1, 2],
            "default": None,
        },
    },
    {
        "name": "prefix",
        "in": "query",
        "description": "Key prefix",
        "schema": OPTIONAL_STRING_SCHEMA,
    },
    {
        "name": "delimiter",
        "in": "query",
        "description": "Key delimiter",
        "schema": OPTIONAL_STRING_SCHEMA,
    },
    {
        "name": "max-keys",
        "in": "query",
        "description": "Maximum number of keys to return",
        "schema": {
            "type": "integer",
            "default": 1000,
        },
    },
    {
        "name": "start-after",
        "in": "query",
        "description": "Start after given key. For list-type=2 only.",
        "schema": OPTIONAL_STRING_SCHEMA,
    },
    {
        "name": "continuation-token",
        "in": "query",
        "description": "Continuation token for paging. For list-type=2 only.",
        "schema": OPTIONAL_STRING_SCHEMA,
    },
    {
        "name": "marker",
        "in": "query",
        "description": "Continuation token for paging. For list-type=1 only.",
        "schema": OPTIONAL_STRING_SCHEMA,
    },
]

GET_OBJECT_PARAMETERS = [
    BUCKET_PARAMETER,
    {
        "name": "key",
        "in": "path",
        "description": "The object's key.",
        "schema": {
            "type": "string",
        },
    },
]


@api.route("/s3/{bucket}")
class ListS3BucketHandler(ApiHandler[S3Context]):
    @api.operation(
        operationId="listS3Bucket",
        summary="List bucket contents.",
        parameters=LIST_BUCKET_PARAMETERS,
    )
    async def get(self, bucket: str):
        try:
            object_storage = self.ctx.get_bucket(bucket)
        except KeyError:
            return await self._bucket_not_found(bucket)

        list_s3_bucket_params = dict(
            prefix=self.request.get_query_arg("prefix", default=None),
            delimiter=self.request.get_query_arg("delimiter", default=None),
            max_keys=int(self.request.get_query_arg("max-keys", default="1000")),
        )

        list_type = self.request.get_query_arg("list-type", default=None)
        if list_type == "2":  # Most frequently used
            list_s3_bucket = list_s3_bucket_v2
            list_s3_bucket_params.update(
                start_after=self.request.get_query_arg("start-after", default=None),
                continuation_token=self.request.get_query_arg(
                    "continuation-token", default=None
                ),
            )
        elif list_type in (None, "1"):  # Old clients
            list_s3_bucket = list_s3_bucket_v1
            list_s3_bucket_params.update(
                marker=self.request.get_query_arg("marker", default=None)
            )
        else:
            raise ApiError.BadRequest(f"Unknown bucket list type {list_type!r}")

        list_bucket_result = list_s3_bucket(
            object_storage, name=bucket, **list_s3_bucket_params
        )

        xml = list_bucket_result_to_xml(list_bucket_result)
        self.response.set_header("Content-Type", "application/xml")
        await self.response.finish(xml)


@api.route("/s3/{bucket}/{*key}")
class GetS3BucketObjectHandler(ApiHandler[S3Context]):
    @api.operation(
        operationId="getS3ObjectMetadata",
        summary="Get object metadata for given key.",
        parameters=GET_OBJECT_PARAMETERS,
    )
    async def head(self, bucket: str, key: Optional[str]):
        try:
            object_storage = self.ctx.get_bucket(bucket)
        except KeyError:
            return await self._bucket_not_found(bucket)

        try:
            value = object_storage[key]
        except KeyError:
            return await self._key_not_found(key)

        e_tag = hashlib.md5(value).hexdigest()
        content_length = len(value)

        self.response.set_header("ETag", f'"{e_tag}"')
        self.response.set_header("Last-Modified", _LAST_MODIFIED_DUMMY)
        self.response.set_header("Content-Length", str(content_length))
        await self.response.finish()

    # noinspection PyPep8Naming
    @api.operation(
        operationId="getS3ObjectData",
        summary="Get object for given key.",
        parameters=GET_OBJECT_PARAMETERS,
    )
    async def get(self, bucket: str, key: Optional[str]):
        try:
            object_storage = self.ctx.get_bucket(bucket)
        except KeyError:
            return await self._bucket_not_found(bucket)

        try:
            value = object_storage[key]
        except KeyError:
            return await self._key_not_found(key)

        e_tag = hashlib.md5(value).hexdigest()
        content_length = len(value)

        self.response.set_header("ETag", f'"{e_tag}"')
        self.response.set_header("Last-Modified", _LAST_MODIFIED_DUMMY)
        self.response.set_header("Content-Length", str(content_length))
        self.response.set_header("Content-Type", "binary/octet-stream")
        await self.response.finish(value)

    def _key_not_found(self, key: str):
        return self._not_found(
            "NoSuchKey", "The specified key does not exist.", Key=key
        )

    def _bucket_not_found(self, bucket_name: str):
        return self._not_found(
            "NoSuchBucket",
            "The specified bucket does not exist.",
            BucketName=bucket_name,
        )

    def _not_found(self, code: str, message: str, **kwargs):
        self.response.set_header("Content-Type", "application/xml")
        self.response.set_status(404)
        return self.response.finish(
            dict_to_xml(
                root_element_name="Error",
                content_dict={"Code": code, "Message": message, **kwargs},
            )
        )
