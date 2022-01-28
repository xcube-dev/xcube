# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
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

import io
import uuid
from abc import ABCMeta, abstractmethod
from typing import Tuple, Union, Any, Sequence

import dask.array as da
import numpy as np
from PIL import Image

from xcube.constants import LOG
from xcube.util.assertions import assert_instance
from xcube.util.assertions import assert_true
from xcube.util.cache import Cache
from xcube.util.cmaps import get_cmap
from xcube.util.perf import measure_time_cm

try:
    import cmocean.cm as ocm
except ImportError:
    ocm = None

__author__ = "Norman Fomferra (Brockmann Consult GmbH)"

DEFAULT_COLOR_MAP_NAME = 'viridis'
DEFAULT_COLOR_MAP_VALUE_RANGE = 0.0, 1.0
DEFAULT_COLOR_MAP_NUM_COLORS = 256

Size2D = Tuple[int, int]
Rectangle2D = Tuple[int, int, int, int]
NDArrayLike = Union[da.Array, np.ndarray]
Tile = Any
NormRange = Tuple[Union[int, float], Union[int, float]]


class TiledImage(metaclass=ABCMeta):
    """
    The interface for tiled images.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """
        Return a unique image identifier.
        :return: A unique (string) object
        """

    @property
    @abstractmethod
    def format(self) -> str:
        """
        Return a format string such as 'PNG', 'JPG', 'RAW', etc, or None according to PIL.
        :return: A string indicating the image (file) format.
        """

    @property
    @abstractmethod
    def mode(self) -> str:
        """
        Return the image mode string such as 'RGBA', 'RGB', 'L', etc, or None according to PIL.
        See http://pillow.readthedocs.org/en/3.0.x/handbook/concepts.html#modes
        :return: A string indicating the image mode
        """

    @property
    @abstractmethod
    def size(self) -> Size2D:
        """
        :return: The size of the image as a (width, height) tuple
        """

    @property
    @abstractmethod
    def tile_size(self) -> Size2D:
        """
        :return: The size of the image as a (tile_width, tile_height) tuple
        """

    @property
    @abstractmethod
    def num_tiles(self) -> Size2D:
        """
        :return: The number of tiles as a (num_tiles_x, num_tiles_y) tuple
        """

    @abstractmethod
    def get_tile(self, tile_x, tile_y) -> Tile:
        """
        Get the tile at tile indices *tile_x*, *tile_y*.

        :param tile_x: the tile index in X direction
        :param tile_y: the tile index in Y direction
        :return: The image's tile data at tile_x, tile_y.
        """

    @abstractmethod
    def dispose(self) -> None:
        """
        Dispose resources allocated by this image.
        """


class AbstractTiledImage(TiledImage, metaclass=ABCMeta):
    """
    An abstract base class for tiled images.
    Derived classes must implement the get_tile(tile_x, tile_y) method.
    It is strongly advised to also override the dispose() method in order to release any allocated resources.

    :param size: the image size as (width, height)
    :param tile_size: tile size as (tile_width, tile_height)
    :param num_tiles: number of tiles as (num_tiles_x, num_tiles_y)
    :param mode: optional mode string
    :param format: optional format string
    :param image_id: optional unique image identifier
    """

    # noinspection PyShadowingBuiltins
    def __init__(self,
                 size: Size2D,
                 tile_size: Size2D,
                 num_tiles: Size2D,
                 mode: str = None,
                 format: str = None,
                 image_id: str = None):
        self._width = size[0]
        self._height = size[1]
        self._tile_width = tile_size[0]
        self._tile_height = tile_size[1]
        self._num_tiles_x = num_tiles[0]
        self._num_tiles_y = num_tiles[1]
        self._id = image_id or str(uuid.uuid4())
        self._mode = mode
        self._format = format

    @property
    def id(self) -> str:
        return self._id

    @property
    def format(self) -> str:
        return self._format

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def size(self) -> Size2D:
        return self._width, self._height

    @property
    def tile_size(self) -> Size2D:
        return self._tile_width, self._tile_height

    @property
    def num_tiles(self) -> Size2D:
        return self._num_tiles_x, self._num_tiles_y

    def dispose(self) -> None:
        """
        Does nothing.
        """

    def get_tile_id(self, tile_x, tile_y):
        return '%s/%d/%d' % (self.id, tile_x, tile_y)


class OpImage(AbstractTiledImage, metaclass=ABCMeta):
    """
    An abstract base class for images that compute their tiles.
    Derived classes must implement the compute_tile(tile_x, tile_y, rectangle) method only.

    :param size: the image size as (width, height)
    :param tile_size: optional tile size as (tile_width, tile_height)
    :param num_tiles: optional number of tiles as (num_tiles_x, num_tiles_y)
    :param mode: optional mode string
    :param format: optional format string
    :param image_id: optional unique image identifier
    :param tile_cache: optional tile cache
    :param trace_perf: whether to trace runtime performance information
    """

    # noinspection PyShadowingBuiltins
    def __init__(self,
                 size: Size2D,
                 tile_size: Size2D,
                 num_tiles: Size2D,
                 mode: str = None,
                 format: str = None,
                 image_id: str = None,
                 tile_cache: Cache = None,
                 trace_perf: bool = False):
        super().__init__(size, tile_size, num_tiles, mode=mode, format=format, image_id=image_id)
        self._tile_cache = tile_cache
        self._trace_perf = trace_perf

    @property
    def tile_cache(self) -> Cache:
        return self._tile_cache

    def get_tile(self, tile_x: int, tile_y: int) -> Tile:

        measure_time = self.measure_time
        tile_id = self.get_tile_id(tile_x, tile_y)
        tile_tag = self.__get_tile_tag(tile_id)
        tile_cache = self._tile_cache

        if tile_cache:
            with measure_time(tile_tag + 'queried in tile cache'):
                tile = tile_cache.get_value(tile_id)
            if tile is not None:
                if self._trace_perf:
                    LOG.info(tile_tag + 'restored from tile cache')
                return tile

        with measure_time(tile_tag + 'computed'):
            tw, th = self.tile_size
            tile = self.compute_tile(tile_x, tile_y, (tw * tile_x, th * tile_y, tw, th))

        if tile_cache:
            with measure_time(tile_tag + 'stored in tile cache'):
                tile_cache.put_value(tile_id, tile)

        return tile

    @abstractmethod
    def compute_tile(self, tile_x: int, tile_y: int, rectangle: Rectangle2D) -> Tile:
        """
        Compute a tile at tile indices *tile_x*, *tile_y*.
        The tile's boundaries are provided in *rectangle* given in image pixel coordinates.

        :param tile_x: the tile index in X direction
        :param tile_y: the tile index in Y direction
        :param rectangle: tile rectangle is given in image pixel coordinates.
        :return: a new tile
        """

    def dispose(self) -> None:
        cache = self._tile_cache
        if cache:
            num_tiles_x, num_tiles_y = self.num_tiles
            for tile_y in range(num_tiles_y):
                for tile_x in range(num_tiles_x):
                    cache.remove_value(self.get_tile_id(tile_x, tile_y))

    @property
    def measure_time(self):
        """ A context manager to measure execution time of code blocks. """
        return measure_time_cm(disabled=not self._trace_perf)

    def _get_tile_tag(self, tile_x: int, tile_y: int) -> str:
        """ Use to log tile computation info """
        return self.__get_tile_tag(self.get_tile_id(tile_x, tile_y))

    @staticmethod
    def __get_tile_tag(tile_id: str) -> str:
        """ Use to log tile computation info """
        return "tile " + tile_id + ": "


class DecoratorImage(OpImage, metaclass=ABCMeta):
    """
    Abstract tiled image class allowing behavior to be added to a given tiled source image.
    The decorator image will have the same image layout as the source image.
    Derived classes must implement the compute_tile_from_source_tile() method only.

    :param source_image: the source image
    :param image_id: optional unique image identifier
    :param format: optional format string
    :param mode: optional mode string
    :param tile_cache: optional tile cache
    :param trace_perf: whether to log runtime performance information
    """

    # noinspection PyShadowingBuiltins
    def __init__(self,
                 source_image: TiledImage,
                 image_id: str = None,
                 format: str = None,
                 mode: str = None,
                 tile_cache: Cache = None,
                 trace_perf: bool = False):
        super().__init__(source_image.size,
                         source_image.tile_size,
                         source_image.num_tiles,
                         mode=mode if mode else source_image.mode,
                         format=format if format else source_image.format,
                         image_id=image_id,
                         tile_cache=tile_cache,
                         trace_perf=trace_perf)
        self._source_image = source_image

    @property
    def source_image(self):
        return self._source_image

    def compute_tile(self, tile_x: int, tile_y: int, rectangle: Rectangle2D) -> Tile:
        source_tile = self._source_image.get_tile(tile_x, tile_y)
        target_tile = None
        if source_tile is not None:
            target_tile = self.compute_tile_from_source_tile(tile_x, tile_y, rectangle, source_tile)
        return target_tile

    @abstractmethod
    def compute_tile_from_source_tile(self,
                                      tile_x: int, tile_y: int,
                                      rectangle: Rectangle2D,
                                      source_tile: Tile) -> Tile:
        """
        Compute a tile from the given *source_tile*.

        :param tile_x: the tile index in X direction
        :param tile_y: the tile index in Y direction
        :param rectangle: tile rectangle is given in image pixel coordinates.
        :param source_tile: the source tile
        :return: a new tile computed from the source tile
        """


class NormalizeArrayImage(DecoratorImage):
    """
    Performs basic (numpy) array tile transformations.
    Currently available: norm_range.
    Expects the source image to provide (numpy) arrays.

    :param source_image: The source image
    :param image_id: Optional unique image identifier
    :param tile_cache: Optional tile cache
    :param trace_perf: Whether to log runtime performance information
    """

    def __init__(self,
                 source_image: TiledImage,
                 image_id: str = None,
                 force_2d: bool = False,
                 norm_range: NormRange = None,
                 tile_cache: Cache = None,
                 trace_perf: bool = False):
        super().__init__(source_image, image_id=image_id, tile_cache=tile_cache, trace_perf=trace_perf)
        self._force_2d = force_2d
        self._norm_range = norm_range

    def compute_tile(self, tile_x: int, tile_y: int, rectangle: Rectangle2D) -> Tile:
        source_tile = self._source_image.get_tile(tile_x, tile_y)
        target_tile = None
        if source_tile is not None:
            # noinspection PyTypeChecker
            target_tile = self.compute_tile_from_source_tile(tile_x, tile_y, rectangle, source_tile)
        return target_tile

    def compute_tile_from_source_tile(self, tile_x: int, tile_y: int, rectangle: Rectangle2D, tile: Tile) -> Tile:
        measure_time = self.measure_time
        tile_tag = self._get_tile_tag(tile_x, tile_y)

        if self._norm_range is not None:
            norm_min, norm_max = self._norm_range
            with measure_time(tile_tag + "normalize_min_max"):
                tile = np.clip(tile, norm_min, norm_max)
                tile -= norm_min
                tile *= 1.0 / (norm_max - norm_min)

        return tile


class DirectRgbaImage(OpImage):
    """
    Creates an RGBA image from a three source images that provide tiles as normalized, numpy-like 2D arrays.

    :param source_images: the source images
    :param image_id: optional unique image identifier
    :param encode: Whether to create tiles that are encoded image bytes according to *format*.
    :param format: Image format, e.g. "JPEG", "PNG"
    :param tile_cache: optional tile cache
    :param trace_perf: whether to log runtime performance information
    """

    # noinspection PyShadowingBuiltins
    def __init__(self,
                 source_images: Sequence[TiledImage],
                 image_id: str = None,
                 encode: bool = False,
                 format: str = None,
                 tile_cache: Cache = None,
                 trace_perf: bool = False):
        assert_instance(source_images, (list, tuple),
                        name='source_images')
        assert_true(len(source_images) == 3,
                    message='source_images must have length 3')
        proto_source_image = source_images[0]
        super().__init__(size=proto_source_image.size,
                         tile_size=proto_source_image.tile_size,
                         num_tiles=proto_source_image.num_tiles,
                         image_id=image_id,
                         format=format,
                         mode='RGBA',
                         tile_cache=tile_cache,
                         trace_perf=trace_perf)
        self._source_images = tuple(source_images)
        self._encode = encode

    def compute_tile(self, tile_x: int, tile_y: int, rectangle: Rectangle2D) -> Tile:

        measure_time = self.measure_time
        tile_tag = self._get_tile_tag(tile_x, tile_y)

        with measure_time(tile_tag + "get tiles"):
            imr, img, imb = self._source_images
            tr = imr.get_tile(tile_x, tile_y)
            tg = img.get_tile(tile_x, tile_y)
            tb = imb.get_tile(tile_x, tile_y)

        w, h = self.tile_size
        with measure_time(tile_tag + "construct rgba array"):
            tile = np.zeros((h, w, 4), dtype=np.uint8)
            tile[..., 0] = tr * 255.9999
            tile[..., 1] = tg * 255.9999
            tile[..., 2] = tb * 255.9999
            tile[..., 3] = np.where(np.isfinite(tr + tg + tb), 255, 0)

        with measure_time(tile_tag + "create image"):
            image = Image.fromarray(tile, mode=self.mode)

        if self._encode and self.format:
            with measure_time(tile_tag + "encode PNG"):
                ostream = io.BytesIO()
                image.save(ostream, format=self.format)
                encoded_image = ostream.getvalue()
                ostream.close()
                return encoded_image
        else:
            return image


class ColorMappedRgbaImage(DecoratorImage):
    """
    Creates a color-mapped image from a source image that provide tiles as numpy-like image arrays.

    :param source_image: the source image
    :param image_id: optional unique image identifier
    :param cmap_name: A Matplotlib color map name
    :param num_colors: Number of colors
    :param encode: Whether to create tiles that are encoded image bytes according to *format*.
    :param format: Image format, e.g. "JPEG", "PNG"
    :param tile_cache: optional tile cache
    :param trace_perf: whether to log runtime performance information
    """

    # noinspection PyShadowingBuiltins
    def __init__(self,
                 source_image: TiledImage,
                 image_id: str = None,
                 cmap_name: str = DEFAULT_COLOR_MAP_NAME,
                 num_colors: int = DEFAULT_COLOR_MAP_NUM_COLORS,
                 encode: bool = False,
                 format: str = None,
                 tile_cache: Cache = None,
                 trace_perf: bool = False):
        super().__init__(source_image, image_id=image_id, format=format, mode='RGBA', tile_cache=tile_cache,
                         trace_perf=trace_perf)
        cmap_name, cmap = get_cmap(cmap_name or DEFAULT_COLOR_MAP_NAME,
                                   default_cmap_name=DEFAULT_COLOR_MAP_NAME,
                                   num_colors=num_colors or DEFAULT_COLOR_MAP_NUM_COLORS)
        self._cmap_name = cmap_name
        self._cmap = cmap
        self._encode = encode

    def compute_tile_from_source_tile(self,
                                      tile_x: int, tile_y: int,
                                      rectangle: Rectangle2D, source_tile: Tile) -> Tile:
        measure_time = self.measure_time
        tile_tag = self._get_tile_tag(tile_x, tile_y)

        tile = source_tile
        with measure_time(tile_tag + "map colors"):
            tile = self._cmap(tile, bytes=True)

        with measure_time(tile_tag + "create image"):
            image = Image.fromarray(tile, mode=self.mode)

        if self._encode and self.format:
            with measure_time(tile_tag + "encode PNG"):
                ostream = io.BytesIO()
                image.save(ostream, format=self.format)
                encoded_image = ostream.getvalue()
                ostream.close()
                return encoded_image
        else:
            return image


class SourceArrayImage(OpImage):
    """
    A tiled image created from a numpy ndarray-like data array.

    :param array: a numpy-ndarray-like data array
    :param tile_size: the tile size
    :param image_id: optional unique image identifier
    :param flip_y: Whether to flip pixels in y-direction
    :param tile_cache: an optional tile cache
    """

    def __init__(self,
                 array: Union[NDArrayLike, Any],
                 tile_size: Size2D,
                 image_id: str = None,
                 flip_y: bool = False,
                 tile_cache: Cache = None,
                 trace_perf: bool = False):
        if len(array.shape) != 2:
            raise ValueError('array must be 2D')
        width, height = array.shape[-1], array.shape[-2]
        if width <= 0 or height <= 0:
            raise ValueError('array sizes must be positive')
        tile_width, tile_height = tile_size
        if tile_width <= 0 or tile_width <= 0:
            raise ValueError('tile sizes must be positive')
        num_tiles = (width + tile_width - 1) // tile_width, (height + tile_height - 1) // tile_height
        super().__init__((width, height),
                         tile_size=tile_size,
                         num_tiles=num_tiles,
                         mode=str(array.dtype),
                         image_id=image_id,
                         tile_cache=tile_cache,
                         trace_perf=trace_perf)
        is_xarray_like = hasattr(array, 'data') and hasattr(array, 'dims') and hasattr(array, 'attrs')
        self._array = array.data if is_xarray_like else array
        self._flip_y = flip_y
        self._tile_offset_y = self.size[1] % self.tile_size[1]

    def compute_tile(self, tile_x: int, tile_y: int, rectangle: Rectangle2D) \
            -> NDArrayLike:
        measure_time = self.measure_time
        tile_tag = self._get_tile_tag(tile_x, tile_y)
        x, y, w, h = rectangle

        if self._flip_y:
            num_tiles_y = self.num_tiles[1]
            tile_size_y = self.tile_size[1]
            tile_y = num_tiles_y - 1 - tile_y
            if self._tile_offset_y > 0:
                if tile_y == 0:
                    y = 0
                    h = self._tile_offset_y
                else:
                    y = self._tile_offset_y + (tile_y - 1) * tile_size_y
            else:
                y = tile_y * tile_size_y

        tile = self._array[y:y + h, x:x + w]
        if self._flip_y:
            with measure_time(tile_tag + "flip y"):
                # Flip tile using fancy indexing
                tile = tile[..., ::-1, :]
        # ensure that our tile size is w x h
        return trim_tile(tile, self.tile_size)


def trim_tile(tile: NDArrayLike,
              expected_tile_size: Size2D,
              fill_value: float = np.nan) -> NDArrayLike:
    """
    Trim a tile.

    If too small, expand and pad with background value. If too large, crop.

    :param tile: The tile
    :param expected_tile_size: expected tile size
    :param fill_value: fill value for padding
    :return: the trimmed tile
    """
    expected_width, expected_height = expected_tile_size
    actual_width, actual_height = tile.shape[-1], tile.shape[-2]
    if expected_width > actual_width:
        # expand in width and pad with fill_value
        h_pad = np.empty((actual_height, expected_width - actual_width))
        h_pad.fill(fill_value)
        tile = np.hstack((tile, h_pad))
    if expected_height > actual_height:
        # expand in height and pad with fill_value
        v_pad = np.empty((expected_height - actual_height, expected_width))
        v_pad.fill(fill_value)
        tile = np.vstack((tile, v_pad))
    if expected_width < actual_width or expected_height < actual_height:
        # crop
        tile = tile[0:expected_height, 0:expected_width]
    return tile
