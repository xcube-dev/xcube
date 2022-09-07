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

import os.path
import pathlib
from typing import Optional

from xcube.constants import LOG
from xcube.server.api import ApiError
from xcube.server.api import ApiHandler
from xcube.webapi.s3.s3util import (
    dict_to_xml,
    list_s3_bucket_v1,
    list_bucket_result_to_xml,
    list_s3_bucket_v2,
    mtime_to_str,
    str_to_etag)
from .context import S3Context
from .api import api

_LOG_S3BUCKET_HANDLER = False


@api.route('/s3bucket')
class ListS3BucketHandler(ApiHandler[S3Context]):

    @api.operation(operationId='listS3Bucket',
                   summary='List datasets accessible through the S3 API.')
    async def get(self):

        prefix = self.request.get_query_arg('prefix', default=None)
        delimiter = self.request.get_query_arg('delimiter', default=None)
        max_keys = int(self.request.get_query_arg('max-keys', default='1000'))
        list_s3_bucket_params = dict(prefix=prefix, delimiter=delimiter,
                                     max_keys=max_keys)

        list_type = self.request.get_query_arg('list-type', default=None)
        if list_type is None:
            marker = self.request.get_query_arg('marker', default=None)
            list_s3_bucket_params.update(marker=marker)
            list_s3_bucket = list_s3_bucket_v1
        elif list_type == '2':
            start_after = self.request.get_query_arg(
                'start-after', default=None
            )
            continuation_token = self.request.get_query_arg(
                'continuation-token', default=None
            )
            list_s3_bucket_params.update(
                start_after=start_after,
                continuation_token=continuation_token
            )
            list_s3_bucket = list_s3_bucket_v2
        else:
            raise ApiError.BadRequest(
                f'Unknown bucket list type {list_type!r}')

        if _LOG_S3BUCKET_HANDLER:
            LOG.info(f'GET: list_s3_bucket_params={list_s3_bucket_params}')
        bucket_mapping = self.ctx.get_s3_bucket_mapping()
        list_bucket_result = list_s3_bucket(bucket_mapping,
                                            **list_s3_bucket_params)
        if _LOG_S3BUCKET_HANDLER:
            import json
            LOG.info(f'-->\n{json.dumps(list_bucket_result, indent=2)}')

        xml = list_bucket_result_to_xml(list_bucket_result)
        self.response.set_header('Content-Type', 'application/xml')
        await self.response.finish(xml)


@api.route('/s3bucket/{datasetId}/{*path}')
class GetS3BucketObjectHandler(ApiHandler[S3Context]):
    # noinspection PyPep8Naming
    @api.operation(operationId='getS3ObjectMetadata',
                   summary='Get the metadata for a dataset S3 object.')
    async def head(self, datasetId: str, path: Optional[str]):
        path = path or ""
        key, local_path = self._get_key_and_local_path(datasetId, path)
        if _LOG_S3BUCKET_HANDLER:
            LOG.info(f'HEAD: key={key!r}, local_path={local_path!r}')
        if local_path is None or not local_path.exists():
            await self._key_not_found(key)
            return
        self.response.set_header('ETag', str_to_etag(str(local_path)))
        self.response.set_header('Last-Modified',
                                 mtime_to_str(local_path.stat().st_mtime))
        if local_path.is_file():
            self.response.set_header('Content-Length',
                                     str(local_path.stat().st_size))
        else:
            self.response.set_header('Content-Length', '0')
        await self.response.finish()

    # noinspection PyPep8Naming
    @api.operation(operationId='getS3ObjectData',
                   summary='Get the data for a dataset S3 object.')
    async def get(self, datasetId: str, path: Optional[str]):
        path = path or ""
        key, local_path = self._get_key_and_local_path(datasetId, path)
        if _LOG_S3BUCKET_HANDLER:
            LOG.info(f'GET: key={key!r}, local_path={local_path!r}')
        if local_path is None or not local_path.exists():
            await self._key_not_found(key)
            return
        self.response.set_header('ETag', str_to_etag(str(local_path)))
        self.response.set_header('Last-Modified',
                                 mtime_to_str(local_path.stat().st_mtime))
        self.response.set_header('Content-Type', 'binary/octet-stream')
        if local_path.is_file():
            self.response.set_header('Content-Length',
                                     str(local_path.stat().st_size))
            chunk_size = 1024 * 1024
            with open(str(local_path), 'rb') as fp:
                while True:
                    chunk = fp.read(chunk_size)
                    if len(chunk) == 0:
                        break
                    await self.response.finish(chunk)
        else:
            self.response.set_header('Content-Length', '0')
            await self.response.finish()

    def _key_not_found(self, key: str):
        self.response.set_header('Content-Type', 'application/xml')
        self.response.set_status(404)
        return self.response.finish(dict_to_xml(
            'Error',
            dict(Code='NoSuchKey',
                 Message='The specified key does not exist.',
                 Key=key))
        )

    def _get_key_and_local_path(self, ds_id: str, path: str):
        dataset_config = self.ctx.datasets_ctx.get_dataset_config(ds_id)
        file_system = dataset_config.get('FileSystem', 'file')
        required_file_systems = ['file', 'local']
        if file_system not in required_file_systems:
            required_file_system_string = " or ".join(required_file_systems)
            raise ApiError.BadRequest(
                f'AWS S3 data access: currently,'
                f' only datasets in file systems'
                f' {required_file_system_string!r} are supported,'
                f' but dataset'
                f' {ds_id!r} uses file system {file_system!r}')

        key = f'{ds_id}/{path}'

        # validate path
        if path and '..' in path.split('/'):
            raise ApiError.BadRequest(
                f'AWS S3 data access: received illegal key {key!r}'
            )

        bucket_mapping = self.ctx.get_s3_bucket_mapping()
        local_path = bucket_mapping.get(ds_id)
        local_path = os.path.join(local_path, path)

        local_path = os.path.normpath(local_path)

        return key, pathlib.Path(local_path)
