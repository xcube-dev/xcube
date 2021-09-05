from abc import abstractmethod, ABC
from typing import Tuple, Union
from unittest import TestCase

from xcube.util.assertions import assert_true


class TileGrid2(ABC):
    @abstractmethod
    def get_tile_size(self) -> Tuple[int, int]:
        """Get the tile size in pixels."""

    @abstractmethod
    def get_num_tiles(self, level: int) -> Tuple[int, int]:
        """Get the number of tiles at *level*"""

    @abstractmethod
    def get_resolution(self, level: int) -> float:
        """Get the spatial resolution at *level*."""

    @abstractmethod
    def get_rectangle(self, level: int) -> Tuple[float, float, float, float]:
        """Get the spatial resolution at *level*."""


class GlobalCrs84TileGrid(TileGrid2):
    def __init__(self,
                 tile_size: Union[int, Tuple[int, int]] = 256):
        if not tile_size:
            tile_size = 256
        elif not isinstance(tile_size, int):
            tile_with, tile_height = tile_size
            assert_true(tile_with == tile_height,
                        message='tile_with, tile_height must be equal')
            tile_size = tile_with
        assert_true(tile_size >= 2, message='tile_size must be >= 2')
        self._tile_size: int = tile_size
        self._res_0 = 180.0 / tile_size
        self._rectangle = -180, -90, 180, 90

    def get_tile_size(self) -> Tuple[int, int]:
        ts = self._tile_size
        return ts, ts

    def get_num_tiles(self, level: int) -> Tuple[int, int]:
        factor = 1 << level
        return 2 * factor, factor

    def get_resolution(self, level: int) -> float:
        factor = 1 << level
        return self._res_0 / factor

    def get_rectangle(self, level: int) -> Tuple[float, float, float, float]:
        return self._rectangle


class ArbitraryTileGrid(TileGrid2):
    def __init__(self,
                 image_size: Tuple[int, int],
                 tile_size: Tuple[int, int],
                 xy_min: Tuple[float, float],
                 xy_res: float):
        image_width, image_height = image_size
        assert_true(image_width >= 2 and image_height >= 2,
                    message='image_width and image_height must be >= 2')
        self._image_width = int(image_width)
        self._image_height = int(image_height)

        tile_width, tile_height = tile_size
        assert_true(tile_width >= 2 and tile_height >= 2,
                    message='tile_width and tile_height must be >= 2')
        tile_width = min(image_width, tile_width)
        tile_height = min(image_height, tile_height)
        self._tile_width = int(tile_width)
        self._tile_height = int(tile_height)

        x_min, y_min = xy_min
        self._xy_min = x_min, y_min
        assert_true(xy_res > 0,
                    message='xy_res must be > 0')

        self._xy_res = xy_res

        num_levels = 1
        image_width_0 = image_width
        image_height_0 = image_height
        while True:
            image_width_0 //= 2
            image_height_0 //= 2
            if image_width_0 <= tile_width or image_height_0 <= tile_height:
                break
            num_levels += 1
        num_level_0_tiles_x = (image_width_0 + tile_width - 1) // tile_width
        num_level_0_tiles_y = (image_height_0 + tile_height - 1) // tile_height

    def get_tile_size(self) -> Tuple[int, int]:
        return self._tile_width, self._tile_height

    def get_num_tiles(self, level: int) -> Tuple[int, int]:
        image_with, image_height = self._image_width, self._image_height
        tile_width, tile_height = self._tile_width, self._tile_height
        factor = 1 << level
        image_width_level = (image_with + 1) // factor
        image_height_level = (image_height + 1) // factor
        num_tiles_x = (image_width_level + tile_width - 1) // tile_width
        num_tiles_y = (image_height_level + tile_height - 1) // tile_height
        return num_tiles_x, num_tiles_y

    def get_resolution(self, level: int) -> float:
        factor = 1 << level
        return self._res_0 / factor

    def get_rectangle(self, level: int) -> Tuple[float, float, float, float]:
        return self._rectangle


class TilingGrid2Test(TestCase):
    def test_it(self):
        pass
