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
from typing import Callable

from xcube.server.config import get_reverse_url_prefix
from xcube.server.config import get_url_prefix
from xcube.server.config import resolve_config_path


class ResolveConfigPathTest(unittest.TestCase):
    def test_resolve_config_path_path_abs(self):
        base_dir = '/cyanoalert/configs'

        self.assertEqual(
            f'{base_dir}/demo.yaml',
            resolve_config_path(
                dict(base_dir=base_dir),
                f'{base_dir}/demo.yaml'
            )
        )

        self.assertEqual(
            '/cyanoalert/testing/configs/demo.yaml',
            resolve_config_path(
                dict(base_dir=base_dir),
                '/cyanoalert/testing/configs/demo.yaml'
            )
        )

    def test_resolve_config_path_url_abs(self):
        base_dir = 's3://cyanoalert/configs'

        self.assertEqual(
            f'{base_dir}/demo.yaml',
            resolve_config_path(
                dict(base_dir=base_dir),
                f'{base_dir}/demo.yaml'
            )
        )

        self.assertEqual(
            's3://cyanoalert/testing/configs/demo.yaml',
            resolve_config_path(
                dict(base_dir=base_dir),
                's3://cyanoalert/testing/configs/demo.yaml'
            )
        )

    def test_resolve_config_path_path_rel(self):
        base_dir = '/cyanoalert/configs'

        self.assertEqual(
            '/cyanoalert/configs/demo.yaml',
            resolve_config_path(
                dict(base_dir=base_dir),
                f'demo.yaml'
            )
        )

        self.assertEqual(
            '/cyanoalert/configs/testing/demo.yaml',
            resolve_config_path(
                dict(base_dir=base_dir),
                'testing/demo.yaml'
            )
        )

        self.assertEqual(
            '/cyanoalert/configs/demo.yaml',
            resolve_config_path(
                dict(base_dir=base_dir),
                './testing/../demo.yaml'
            )
        )

        self.assertEqual(
            f'/cyanoalert/testing/demo.yaml',
            resolve_config_path(
                dict(base_dir=base_dir),
                '../testing/demo.yaml'
            )
        )

    def test_resolve_config_path_url_rel(self):
        base_dir = 's3://cyanoalert/configs'

        self.assertEqual(
            's3://cyanoalert/configs/demo.yaml',
            resolve_config_path(
                dict(base_dir=base_dir),
                f'demo.yaml'
            )
        )

        self.assertEqual(
            's3://cyanoalert/configs/testing/demo.yaml',
            resolve_config_path(
                dict(base_dir=base_dir),
                'testing/demo.yaml'
            )
        )

        self.assertEqual(
            f's3://cyanoalert/configs/demo.yaml',
            resolve_config_path(
                dict(base_dir=base_dir),
                './testing/../demo.yaml'
            )
        )

        self.assertEqual(
            f's3://cyanoalert/testing/demo.yaml',
            resolve_config_path(
                dict(base_dir=base_dir),
                '../testing/demo.yaml'
            )
        )


class UrlPrefixConfigTest(unittest.TestCase):
    def test_get_url_prefix(self):
        self.assert_url_prefix(get_url_prefix,
                               key='url_prefix')

    def test_get_reverse_url_prefix(self):
        self.assert_url_prefix(get_reverse_url_prefix,
                               key='reverse_url_prefix')

    def assert_url_prefix(self,
                          get_prefix: Callable,
                          key: str):
        self.assertEqual('',
                         get_prefix({}))
        self.assertEqual('',
                         get_prefix({key: ''}))
        self.assertEqual('',
                         get_prefix({key: None}))
        self.assertEqual('',
                         get_prefix({key: '/'}))
        self.assertEqual('',
                         get_prefix({key: '//'}))

        self.assertEqual('/api/v1',
                         get_prefix({key: 'api/v1'}))
        self.assertEqual('/api/v1',
                         get_prefix({key: '/api/v1'}))
        self.assertEqual('/api/v1',
                         get_prefix({key: 'api/v1/'}))
        self.assertEqual('/api/v1',
                         get_prefix({key: '/api/v1/'}))
        self.assertEqual('/api/v1',
                         get_prefix({key: '/api/v1//'}))
        self.assertEqual('/api/v1',
                         get_prefix({key: '//api/v1//'}))
        self.assertEqual('/api/v1',
                         get_prefix({key: '///api/v1//'}))

        self.assertEqual('https://test.com',
                         get_prefix({key: 'https://test.com'}))
        self.assertEqual('https://test.com',
                         get_prefix({key: 'https://test.com/'}))
        self.assertEqual('https://test.com/api',
                         get_prefix({key: 'https://test.com/api'}))
        self.assertEqual('http://test.com/api',
                         get_prefix({key: 'http://test.com/api/'}))
