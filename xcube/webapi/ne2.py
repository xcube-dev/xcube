# The MIT License (MIT)
# Copyright (c) 2018 by the xcube development team and contributors
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

import os

from xcube.webapi.im.tiledimage import AbstractTiledImage, ImagePyramid
from xcube.webapi.im.tilegrid import GLOBAL_GEO_EXTENT
from xcube.webapi.im.tilegrid import TileGrid

__author__ = "Norman Fomferra (Brockmann Consult GmbH)"


class NaturalEarth2Image(AbstractTiledImage):
    """
    A `TiledImage` implementation which provides 'Natural Earth v2' image tiles.

    @author Norman Fomferra
    """

    NUM_LEVELS = 3
    NUM_LEVEL_0_TILES_X = 2
    NUM_LEVEL_0_TILES_Y = 1
    TILE_SIZE = 256

    @staticmethod
    def get_pyramid() -> ImagePyramid:
        """
        Return an instance of a 'Natural Earth v2' image pyramid:
        * global coverage
        * JPEG RGB format
        * 3 levels of detail: 0 to 2
        * tile size: 256 pixels
        * 2 x 1 tiles on level zero
        """
        dir_path = os.path.join(os.path.dirname(__file__), 'res', 'ne2')
        return ImagePyramid(TileGrid(NaturalEarth2Image.NUM_LEVELS,
                                     NaturalEarth2Image.NUM_LEVEL_0_TILES_X,
                                     NaturalEarth2Image.NUM_LEVEL_0_TILES_Y,
                                     NaturalEarth2Image.TILE_SIZE,
                                     NaturalEarth2Image.TILE_SIZE,
                                     GLOBAL_GEO_EXTENT,
                                     inv_y=False),
                            [NaturalEarth2Image(dir_path, level) for level in range(NaturalEarth2Image.NUM_LEVELS)])

    def __init__(self, dir_path, z_index):
        factor = 1 << z_index
        num_tiles_x = factor * NaturalEarth2Image.NUM_LEVEL_0_TILES_X
        num_tiles_y = factor * NaturalEarth2Image.NUM_LEVEL_0_TILES_Y
        tile_size = NaturalEarth2Image.TILE_SIZE
        self._z_index = z_index
        self._base_path = '%s/%d' % (dir_path, z_index)
        super().__init__((num_tiles_x * tile_size, num_tiles_y * tile_size),
                         tile_size=(tile_size, tile_size),
                         num_tiles=(num_tiles_x, num_tiles_y), format='JPEG', mode='RGB')

    def get_tile(self, tile_x, tile_y):
        num_tiles_y = self.num_tiles[1]
        path = '%s/%d/%d.jpg' % (self._base_path, tile_x, num_tiles_y - 1 - tile_y)
        with open(path, 'rb') as fp:
            return fp.read()
