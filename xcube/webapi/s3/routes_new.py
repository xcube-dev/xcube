# The MIT License (MIT)
# Copyright (c) 2022 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

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


@api.route('/s3')
class ListS3BucketHandler(ApiHandler[S3Context]):
    @api.operation(operationId='listS3Bucket',
                   summary='List datasets accessible through the S3 API.')
    async def get(self):
        list_s3_bucket_params = dict(
            prefix=self.request.get_query_arg(
                'prefix', default=None
            ),
            delimiter=self.request.get_query_arg(
                'delimiter', default=None
            ),
            max_keys=int(self.request.get_query_arg(
                'max-keys', default='1000')
            )
        )

        list_type = self.request.get_query_arg('list-type', default=None)
        if list_type in (None, '1'):
            list_s3_bucket = list_s3_bucket_v1
            list_s3_bucket_params.update(
                marker=self.request.get_query_arg(
                    'marker', default=None
                )
            )
        elif list_type == '2':
            list_s3_bucket = list_s3_bucket_v2
            list_s3_bucket_params.update(
                start_after=self.request.get_query_arg(
                    'start-after', default=None
                ),
                continuation_token=self.request.get_query_arg(
                    'continuation-token', default=None
                )
            )
        else:
            raise ApiError.BadRequest(
                f'Unknown bucket list type {list_type!r}'
            )

        list_bucket_result = list_s3_bucket(self.ctx.object_storage,
                                            **list_s3_bucket_params)

        xml = list_bucket_result_to_xml(list_bucket_result)
        self.response.set_header('Content-Type', 'application/xml')
        await self.response.finish(xml)


@api.route('/s3/{datasetId}/{*path}')
class GetS3BucketObjectHandler(ApiHandler[S3Context]):
    # noinspection PyPep8Naming
    @api.operation(operationId='getS3ObjectMetadata_New',
                   summary='Get the metadata for a dataset S3 object.')
    async def head(self, datasetId: str, path: Optional[str]):
        key = f"{datasetId}/{path}"
        try:
            value = self.ctx.object_storage[key]
        except KeyError:
            await self._key_not_found(key)
            return

        e_tag = hashlib.md5(value).hexdigest()
        content_length = len(value)

        self.response.set_header('ETag', f'"{e_tag}"')
        self.response.set_header('Last-Modified', _LAST_MODIFIED_DUMMY)
        self.response.set_header('Content-Length', str(content_length))
        await self.response.finish()

    # noinspection PyPep8Naming
    @api.operation(operationId='getS3ObjectData_New',
                   summary='Get the data for a dataset S3 object.')
    async def get(self, datasetId: str, path: Optional[str]):
        key = f"{datasetId}/{path}"
        try:
            value = self.ctx.object_storage[key]
        except KeyError:
            await self._key_not_found(key)
            return

        e_tag = hashlib.md5(value).hexdigest()
        content_length = len(value)

        self.response.set_header('ETag', f'"{e_tag}"')
        self.response.set_header('Last-Modified', _LAST_MODIFIED_DUMMY)
        self.response.set_header('Content-Length', str(content_length))
        self.response.set_header('Content-Type', 'binary/octet-stream')
        await self.response.finish(value)

    def _key_not_found(self, key: str):
        self.response.set_header('Content-Type', 'application/xml')
        self.response.set_status(404)
        return self.response.finish(dict_to_xml(
            root_element_name="Error",
            content_dict=dict(
                Code="NoSuchKey",
                Message="The specified key does not exist.",
                Key=key,
            )
        ))
