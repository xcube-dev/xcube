# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
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

import math
from abc import abstractmethod, ABC
from typing import Tuple, Optional, Union, Dict, Any

import pyproj
from deprecated import deprecated

from xcube.util.assertions import assert_instance
from xcube.util.assertions import assert_true


@deprecated(version='0.10.x',
            reason='use xcube.util.tilegrid2.TileGrid2 instead')
class TileGrid(ABC):
    """
    This interface defines a tile grid as used by the
    OpenLayers or Cesium UI JavaScript libraries.
    """

    @property
    @abstractmethod
    def tile_size(self) -> Tuple[int, int]:
        """Tile size in pixels."""

    @property
    def min_level(self) -> Optional[int]:
        """The minimum level of detail."""
        return None

    @property
    def max_level(self) -> Optional[int]:
        """The maximum level of detail."""
        return None

    @property
    def num_levels(self) -> Optional[int]:
        """The number of detail levels."""
        return self.max_level + 1 \
            if self.max_level is not None else None

    @property
    @abstractmethod
    def crs(self) -> pyproj.CRS:
        """Spatial CRS."""

    @property
    @abstractmethod
    def extent(self) -> Tuple[float, float, float, float]:
        """Spatial extent in units of the CRS."""

    @property
    @abstractmethod
    def origin(self) -> Tuple[float, float]:
        """Spatial origin in units of the CRS."""

    @property
    @abstractmethod
    def is_j_axis_up(self) -> bool:
        """
        Whether the image's j-axis points up.
        Usually it points down.

        If this method returns True, the *origin* should be
        the lower-left corner rather than the upper-left corner
        of *extend*.
        """

    @abstractmethod
    def get_image_size(self, level: int) -> Tuple[int, int]:
        """Get the image size at given *level* of detail."""

    @abstractmethod
    def get_num_tiles(self, level: int) -> Tuple[int, int]:
        """Get the number of tiles at given *level* of detail."""

    @abstractmethod
    def get_resolution(self, level: int) -> float:
        """Get the spatial resolution at given *level* of detail."""


@deprecated(version='0.10.x',
            reason='use xcube.util.tilegrid2.TileGrid2 instead')
class BaseTileGrid(TileGrid):
    """
    The default implementation of the :class:`TileGrid2`
    interface.

    :param tile_size: Tile width and height,
        given as a tuple of positive integers.
    :param num_level_0_tiles: Number of tiles at level zero,
        the lowest level of detail,
        in x and y direction
        given as a tuple of positive integers.
    :param crs: The spatial coordinate reference system.
    :param max_resolution: The spatial resolution at level zero,
        the lowest level of detail
        given as size in CRS units per pixel.
    :param extent: The extent of the tile grid given as
        (x_min, y_min, x_max, y_max) in units of the spatial CRS.
    :param origin: The origin in units of the CRS units
        where x,y axes of the tile grid meet.
    :param is_j_axis_up: Whether the j-axis of the image points up.
    :param min_level: Optional minimum level of detail.
    :param max_level: Optional maximum level of detail.
    """

    def __init__(self,
                 tile_size: Tuple[int, int],
                 num_level_0_tiles: Tuple[int, int],
                 crs: pyproj.CRS,
                 max_resolution: float,
                 extent: Tuple[float, float, float, float],
                 origin: Optional[Tuple[float, float]] = None,
                 is_j_axis_up: Optional[bool] = None,
                 min_level: Optional[int] = None,
                 max_level: Optional[int] = None):
        tile_width, tile_height = tile_size
        assert_instance(tile_width, int, name='tile_width')
        assert_instance(tile_height, int, name='tile_height')
        num_level_0_tiles_x, num_level_0_tiles_y = num_level_0_tiles
        assert_instance(num_level_0_tiles_x, int, name='num_level_0_tiles_x')
        assert_instance(num_level_0_tiles_y, int, name='num_level_0_tiles_y')
        if isinstance(crs, str):
            crs = pyproj.CRS.from_string(crs)
        assert_instance(crs, pyproj.CRS, name='crs')

        self._tile_size = tile_width, tile_height
        self._num_level_zero_tiles = num_level_0_tiles_x, num_level_0_tiles_y
        self._min_level = min_level
        self._max_level = max_level

        self._crs = crs
        self._max_resolution = max_resolution
        self._extent = tuple(extent)
        if origin is not None:
            self._origin = tuple(origin)
        else:
            self._origin = extent[0], extent[3]
        if is_j_axis_up is not None:
            self._is_j_axis_up = is_j_axis_up
        else:
            self._is_j_axis_up = not math.isclose(self._origin[1],
                                                  self._extent[3])

    @property
    def crs(self) -> pyproj.CRS:
        return self._crs

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        return self._extent

    @property
    def origin(self) -> Tuple[float, float]:
        return self._origin

    @property
    def is_j_axis_up(self) -> bool:
        return self._is_j_axis_up

    @property
    def min_level(self) -> Optional[int]:
        return self._min_level

    @property
    def max_level(self) -> Optional[int]:
        return self._max_level

    @property
    def tile_size(self) -> Tuple[int, int]:
        return self._tile_size

    def get_image_size(self, level: int) -> Tuple[int, int]:
        tile_width, tile_height = self.tile_size
        num_tiles_x, num_tiles_y = self.get_num_tiles(level)
        return num_tiles_x * tile_width, num_tiles_y * tile_height

    def get_num_tiles(self, level: int) -> Tuple[int, int]:
        num_level_0_tiles_x, num_level_0_tiles_y = self._num_level_zero_tiles
        factor = 1 << level
        return num_level_0_tiles_x * factor, num_level_0_tiles_y * factor

    def get_resolution(self, level: int) -> float:
        factor = 1 << level
        return self._max_resolution / factor


GEOGRAPHIC_CRS = pyproj.CRS.from_string('CRS84')
WEB_MERCATOR_CRS = pyproj.CRS.from_string('EPSG:3857')


@deprecated(version='0.10.x',
            reason='use xcube.util.tilegrid2.TileGrid2 instead')
class GeographicTileGrid(BaseTileGrid):
    """
    A global CRS-84 tile grid with 2 horizontal tiles and 1 vertical tile
    at level zero.

    :param tile_size: Tile size, defaults to 256.
    :param min_level: Optional minimum level of detail.
    :param max_level: Optional maximum level of detail.
    """

    def __init__(self,
                 tile_size: int = 256,
                 min_level: Optional[int] = None,
                 max_level: Optional[int] = None):
        """

        :param tile_size:
        :param min_level:
        :param max_level:
        """
        assert_instance(tile_size, int, name='tile_size')
        super().__init__(
            tile_size=(tile_size, tile_size),
            num_level_0_tiles=(2, 1),
            crs=GEOGRAPHIC_CRS,
            max_resolution=180. / tile_size,
            extent=(-180., -90., 180., 90.),
            min_level=min_level,
            max_level=max_level
        )


@deprecated(version='0.10.x',
            reason='use xcube.util.tilegrid2.TileGrid2 instead')
class ImageTileGrid(BaseTileGrid):
    """
    A tile grid that is created from a base image of size *image_size*.

    :param image_size: Size of the base image.
    :param tile_size: Tile size of the base image.
    :param crs: Spatial CRS.
    :param xy_res: Spatial resolution of the base image
        in CRS units per pixel.
    :param xy_min: Origin of the base image
        in CRS units.
    :param is_j_axis_up: Whether j-axis of the image points up.
    """

    def __init__(self,
                 image_size: Tuple[int, int],
                 tile_size: Tuple[int, int],
                 crs: Union[str, pyproj.CRS],
                 xy_res: float,
                 xy_min: Tuple[float, float],
                 is_j_axis_up: bool):
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

        assert_true(xy_res > 0,
                    message='xy_res must be > 0')

        # Find number of detail levels and number of tiles
        # at lowest detail level (level zero).
        # It is found once the number of tiles
        # in either direction becomes 1 after continuously
        # subdividing the image sizes by 2.
        num_levels = 1
        image_level_width = image_width
        image_level_height = image_height
        while True:
            num_level_0_tiles_x = (image_level_width + tile_width - 1) \
                                  // tile_width
            num_level_0_tiles_y = (image_level_height + tile_height - 1) \
                                  // tile_height
            if num_level_0_tiles_x == 1 or num_level_0_tiles_y == 1:
                break
            image_level_width = (image_level_width + 1) // 2
            image_level_height = (image_level_height + 1) // 2
            num_levels += 1

        x_min, y_min = xy_min
        extent = (x_min,
                  y_min,
                  x_min + image_width * xy_res,
                  y_min + image_height * xy_res)

        factor = 1 << (num_levels - 1)
        max_resolution = xy_res * factor

        super().__init__(
            tile_size=(tile_width, tile_height),
            num_level_0_tiles=(num_level_0_tiles_x, num_level_0_tiles_y),
            crs=crs,
            max_resolution=max_resolution,
            extent=extent,
            is_j_axis_up=is_j_axis_up,
            max_level=num_levels - 1
        )

    def get_num_tiles(self, level: int) -> Tuple[int, int]:
        image_width, image_height = self.get_image_size(level)
        tile_width, tile_height = self.tile_size
        return ((image_width + tile_width - 1) // tile_width,
                (image_height + tile_height - 1) // tile_height)

    def get_image_size(self, level: int) -> Tuple[int, int]:
        factor = 1 << (self.max_level - level)
        return self._image_width // factor, self._image_height // factor


@deprecated(version='0.10.x',
            reason='should no longer be used')
def tile_grid_to_ol_xyz_source_options(tile_grid: TileGrid, url: str) \
        -> Optional[Dict[str, Any]]:
    """
    Convert :class:`TileGrid2` instance into options to be used with
    ``ol.source.XYZ(options)`` of OpenLayers 4+.

    See

    * https://openlayers.org/en/latest/apidoc/module-ol_source_XYZ-XYZ.html
    * https://openlayers.org/en/latest/apidoc/module-ol_tilegrid_TileGrid-TileGrid.html
    * https://openlayers.org/en/latest/examples/xyz.html

    :param tile_grid: tile grid
    :param url: source url
    :return: A JSON object that represents the ``options`` to be passed to
        OpenLayers ``ol.source.XYZ(options)``.
    """
    if not tile_grid.crs.is_geographic:
        return None

    options = {
        'url': url,
        # 'projection': tile_grid.crs.srs
        'projection': 'EPSG:4326'
    }

    if tile_grid.max_level is not None:
        tile_grid_options = {
            'extent': list(tile_grid.extent),
            'origin': [tile_grid.extent[0], tile_grid.extent[3]],
            'tileSize': list(tile_grid.tile_size),
            'sizes': [list(tile_grid.get_num_tiles(i))
                      for i in range(tile_grid.max_level + 1)],
            'resolutions': [tile_grid.get_resolution(i)
                            for i in range(tile_grid.max_level + 1)]}
        if tile_grid.min_level is not None:
            tile_grid_options['minZoom'] = tile_grid.min_level
        options['tileGrid'] = tile_grid_options
    else:
        if tile_grid.min_level is not None:
            options['minZoom'] = tile_grid.min_level
        options['tileSize'] = list(tile_grid.tile_size)
        options['maxResolution'] = float(tile_grid.get_resolution(0))

    return options
