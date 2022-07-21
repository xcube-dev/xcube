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

import unittest

from test.mixins import AlmostEqualDeepMixin
from test.webapi.helpers import get_api_ctx
from xcube.webapi.meta.context import MetaContext
from xcube.webapi.meta.controllers import get_service_info


def get_meta_ctx(server_config=None) -> MetaContext:
    return get_api_ctx("meta", MetaContext, server_config)


class MetaControllerTest(unittest.TestCase, AlmostEqualDeepMixin):

    def test_get_service_info(self):
        ctx = get_meta_ctx()
        result = get_service_info(ctx)
        self.assertIsInstance(result, dict)
        self.assertEqual(
            {
                'apis',
                'currentTime',
                'description',
                'name',
                'pid',
                'startTime',
                'updateTime',
                'version',
                'versions'
            },
            set(result.keys())
        )
