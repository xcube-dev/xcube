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
from typing import Union

from test.webapi.helpers import get_api_ctx
from xcube.server.api import ApiError
from xcube.server.api import ServerConfig
from xcube.webapi.tiles.context import TilesContext
from xcube.webapi.tiles.controllers import compute_ml_dataset_tile
from xcube.constants import CRS84


def get_tiles_ctx(
        server_config: Union[str, ServerConfig] = "config.yml"
) -> TilesContext:
    return get_api_ctx("tiles", TilesContext, server_config)


class TilesControllerTest(unittest.TestCase):

    def test_compute_ml_dataset_tile(self):
        ctx = get_tiles_ctx()
        tile = compute_ml_dataset_tile(
            ctx, 'demo', 'conc_tsm', CRS84, '0', '0', '0', {}
        )
        self.assertIsInstance(tile, bytes)

        tile = compute_ml_dataset_tile(
            ctx, 'demo', 'conc_tsm', CRS84, '-20', '0', '0', {}
        )
        self.assertIsInstance(tile, bytes)

    def test_compute_ml_dataset_tile_invalid_dataset(self):
        ctx = get_tiles_ctx()
        with self.assertRaises(ApiError.NotFound) as cm:
            compute_ml_dataset_tile(
                ctx, 'demo-rgb', 'conc_tsm', CRS84, '0', '0', '0', {}
            )
        self.assertEqual('HTTP status 404:'
                         ' Dataset "demo-rgb" not found',
                         f'{cm.exception}')

    def test_compute_ml_dataset_tile_invalid_variable(self):
        ctx = get_tiles_ctx()
        with self.assertRaises(ApiError.NotFound) as cm:
            compute_ml_dataset_tile(
                ctx, 'demo', 'conc_tdi', CRS84, '0', '0', '0', {}
            )
        self.assertEqual('HTTP status 404:'
                         ' Variable "conc_tdi" not found in dataset "demo"',
                         f'{cm.exception}')

    def test_compute_ml_dataset_tile_with_all_params(self):
        ctx = get_tiles_ctx()
        tile = compute_ml_dataset_tile(
            ctx, 'demo', 'conc_tsm', CRS84, '0', '0', '0',
            dict(time='current',
                 cbar='plasma',
                 vmin='0.1',
                 vmax='0.3')
        )
        self.assertIsInstance(tile, bytes)

    def test_compute_ml_dataset_tile_with_time_dim(self):
        ctx = get_tiles_ctx()
        tile = compute_ml_dataset_tile(
            ctx, 'demo', 'conc_tsm', CRS84, '0', '0', '0',
            dict(time='2017-01-26')
        )
        self.assertIsInstance(tile, bytes)

        ctx = get_tiles_ctx()
        tile = compute_ml_dataset_tile(
            ctx, 'demo', 'conc_tsm', CRS84, '0', '0', '0',
            dict(
                time='2017-01-26/2017-01-27')
        )
        self.assertIsInstance(tile, bytes)

        ctx = get_tiles_ctx()
        tile = compute_ml_dataset_tile(
            ctx, 'demo', 'conc_tsm', CRS84, '0', '0', '0',
            dict(time='current')
        )
        self.assertIsInstance(tile, bytes)

    def test_compute_ml_dataset_tile_with_invalid_time_dim(self):
        ctx = get_tiles_ctx()
        with self.assertRaises(ApiError.BadRequest) as cm:
            compute_ml_dataset_tile(
                ctx, 'demo', 'conc_tsm', CRS84, '0', '0', '0',
                dict(time='Gnaaark!')
            )
        self.assertEqual(
            "HTTP status 400:"
            " Illegal label 'Gnaaark!' for dimension 'time'",
            f'{cm.exception}')

    def test_get_dataset_rgb_tile(self):
        ctx = get_tiles_ctx('config-rgb.yml')
        tile = compute_ml_dataset_tile(
            ctx, 'demo-rgb', 'rgb', CRS84, '0', '0', '0', {}
        )
        self.assertIsInstance(tile, bytes)

    def test_get_dataset_rgb_tile_invalid_b(self):
        ctx = get_tiles_ctx('config-rgb.yml')
        with self.assertRaises(ApiError.NotFound) as cm:
            compute_ml_dataset_tile(
                ctx, 'demo-rgb', 'rgb', CRS84, '0', '0', '0',
                dict(b='refl_3')
            )
        self.assertEqual("HTTP status 404:"
                         " Variable 'refl_3' not found in dataset 'demo-rgb'",
                         f'{cm.exception}')

    def test_get_dataset_rgb_tile_no_vars(self):
        ctx = get_tiles_ctx()
        with self.assertRaises(ApiError.BadRequest) as cm:
            compute_ml_dataset_tile(
                ctx, 'demo', 'rgb', CRS84, '0', '0', '0', {}
            )
        self.assertEqual("HTTP status 400:"
                         " No variable in dataset 'demo'"
                         " specified for RGB component R",
                         f'{cm.exception}')
