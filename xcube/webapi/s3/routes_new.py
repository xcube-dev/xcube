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

import datetime
import hashlib
import time
from typing import Optional

from xcube.server.api import ApiHandler
from .api import api
from .context import S3Context


@api.route('/s3/{datasetId}/{*path}')
class GetS3BucketObjectHandler(ApiHandler[S3Context]):
    _TIME = time.time()

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
        last_modified = datetime.datetime.utcfromtimestamp(self._TIME)
        content_length = len(value)

        self.response.set_header('ETag', f'"{e_tag}"')
        self.response.set_header('Last-Modified', str(last_modified))
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
        last_modified = datetime.datetime.utcfromtimestamp(self._TIME)
        content_length = len(value)

        self.response.set_header('ETag', f'"{e_tag}"')
        self.response.set_header('Last-Modified', str(last_modified))
        self.response.set_header('Content-Length', str(content_length))
        self.response.set_header('Content-Type', 'binary/octet-stream')
        await self.response.finish(value)

    def _key_not_found(self, key: str):
        self.response.set_header('Content-Type', 'application/xml')
        self.response.set_status(404)
        return self.response.finish(
            f"<Error>"
            f"  <Code>NoSuchKey</Code>"
            f"  <Message>The specified key does not exist.</Message>"
            f"  <Key>{key}</Key>"
            f"</Error>"
        )
