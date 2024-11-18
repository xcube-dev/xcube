# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import math
from typing import Optional, Tuple, List, Union
from collections.abc import Sequence

import numpy as np
import pyproj

from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_instance
from xcube.util.types import Pair
from xcube.util.types import ScalarOrPair
from xcube.util.types import normalize_scalar_or_pair

WEB_MERCATOR_CRS_NAME = "EPSG:3857"
WEB_MERCATOR_CRS_ALIASES = (
    "EPSG:3857",
    "urn:ogc:def:crs:OGC:1.3:EPSG::3857",
)
GEOGRAPHIC_CRS_NAME = "OGC:CRS84"
GEOGRAPHIC_CRS_ALIASES = (
    "EPSG:4326",
    "urn:ogc:def:crs:OGC:1.3:EPSG::4326",
    "OGC:CRS84",
    "CRS84",
    "urn:ogc:def:crs:OGC:1.3:CRS84",
    "WGS84",
)
DEFAULT_CRS_NAME = GEOGRAPHIC_CRS_NAME
DEFAULT_TILE_SIZE = 256
EARTH_EQUATORIAL_RADIUS_WGS84 = 6378137.0
EARTH_CIRCUMFERENCE_WGS84 = 2 * math.pi * EARTH_EQUATORIAL_RADIUS_WGS84


class TilingScheme:
    """A scheme for subdividing the world into a tiled, multi-resolution
    image pyramid.

    Two standard tiling schemes are pre-defined:

    1. TilingScheme.WEB_MERCATOR
    2. TilingScheme.GEOGRAPHIC

    Args:
        num_level_zero_tiles: The number of tiles in x and y direction
            at lowest resolution level (zero).
        crs_name: Name of the spatial coordinate reference system.
        map_height: The height of the world map in units of the spatial
            coordinate reference system.
        tile_size: The tile size to be used for both x and y directions.
            Defaults to 256.
        min_level: Optional minimum resolution level.
        max_level: Optional maximum resolution level.
    """

    # The web mercator tiling schema with 1 x 1 level zero tiles
    WEB_MERCATOR: "TilingScheme"
    # The geographic tiling schema with 2 x 1 level zero tiles
    GEOGRAPHIC: "TilingScheme"

    def __init__(
        self,
        num_level_zero_tiles: tuple[int, int],
        crs_name: str,
        map_height: float,
        tile_size: ScalarOrPair[int] = DEFAULT_TILE_SIZE,
        min_level: Optional[int] = None,
        max_level: Optional[int] = None,
    ):
        tile_height, tile_width = normalize_scalar_or_pair(tile_size)
        self._num_level_zero_tiles = num_level_zero_tiles
        self._crs_name = crs_name
        self._crs = None
        self._map_height = map_height
        self._tile_width = tile_width
        self._tile_height = tile_height
        self._min_level = min_level
        self._max_level = max_level

    @property
    def num_level_zero_tiles(self) -> Pair[int]:
        """The number of level zero tiles in x and y directions."""
        return self._num_level_zero_tiles

    @property
    def level_zero_resolution(self) -> float:
        """The (vertical) resolution at level zero in map units."""
        return self._map_height / self._tile_height

    @property
    def crs_name(self) -> str:
        """The name of the spatial coordinate reference system."""
        return self._crs_name

    @property
    def crs(self) -> pyproj.CRS:
        """The spatial coordinate reference system."""
        if self._crs is None:
            self._crs = pyproj.CRS.from_string(self.crs_name)
        return self._crs

    @property
    def map_unit_name(self) -> str:
        """The name of the map's spatial units."""
        return self.crs.axis_info[0].unit_name

    @property
    def map_width(self) -> float:
        """The height of the map in units
        of the map's spatial coordinate reference system.
        """
        num_tiles_x0, num_tiles_y0 = self.num_level_zero_tiles
        return self.map_height * num_tiles_x0 / num_tiles_y0

    @property
    def map_height(self) -> float:
        """The height of the map in units
        of the map's spatial coordinate reference system.
        """
        return self._map_height

    @property
    def map_extent(self) -> tuple[float, float, float, float]:
        """The extent of the map in units
        of the map's spatial coordinate reference system.
        """
        map_width_05 = self.map_width / 2
        map_height_05 = self.map_height / 2
        return -map_width_05, -map_height_05, map_width_05, map_height_05

    @property
    def map_origin(self) -> tuple[float, float]:
        """The origin of the map (upper, left pixel of the upper left tile)
        in units of the map's spatial coordinate reference system.
        """
        map_extent = self.map_extent
        return map_extent[0], map_extent[3]

    @property
    def tile_size(self) -> Pair[int]:
        """The tile size in pixels."""
        return self._tile_width, self._tile_height

    @property
    def tile_width(self) -> int:
        """The tile width in pixels."""
        return self._tile_width

    @property
    def tile_height(self) -> int:
        """The tile height in pixels."""
        return self._tile_height

    @property
    def min_level(self) -> Optional[int]:
        """The minimum level of detail."""
        return self._min_level

    @property
    def max_level(self) -> Optional[int]:
        """The maximum level of detail."""
        return self._max_level

    @property
    def num_levels(self) -> Optional[int]:
        """The number of detail levels."""
        return self.max_level + 1 if self.max_level is not None else None

    @classmethod
    def for_crs(cls, crs: Union[pyproj.CRS, str], **kwargs) -> "TilingScheme":
        """Get a new tiling scheme for the named coordinate reference system.

        Args:
            crs: The spatial coordinate reference system.
            **kwargs: subset of constructor arguments

        Returns:
            a tiling scheme
        """
        assert_instance(crs, (pyproj.CRS, str), name="crs")
        crs_name = crs.srs if isinstance(crs, pyproj.CRS) else crs
        if crs_name in WEB_MERCATOR_CRS_ALIASES:
            tiling_scheme = cls.WEB_MERCATOR
        elif crs_name in GEOGRAPHIC_CRS_ALIASES:
            tiling_scheme = cls.GEOGRAPHIC
        else:
            raise ValueError(f"unsupported spatial CRS {crs_name!r}")
        return tiling_scheme.derive(**kwargs) if kwargs else tiling_scheme

    def derive(self, **kwargs):
        """Derive a new tiling scheme using a subset of constructor
        arguments *kwargs*.

        Args:
            **kwargs: subset of constructor arguments

        Returns:
            a new tiling scheme
        """
        args = self.to_dict()
        args.update({k: v for k, v in kwargs.items() if v is not None})
        return TilingScheme(**args)

    def to_dict(self):
        """Return a JSON-serializable dictionary
        representation of this tiling scheme.
        """
        d = dict(
            num_level_zero_tiles=self.num_level_zero_tiles,
            crs_name=self.crs_name,
            map_height=self.map_height,
            tile_size=self.tile_size,
            min_level=self.min_level,
            max_level=self.max_level,
        )
        return {k: v for k, v in d.items() if v is not None}

    def get_resolutions(
        self,
        min_level: Optional[int] = None,
        max_level: Optional[int] = None,
        unit_name: Optional[str] = None,
    ) -> list[float]:
        """Get spatial resolutions for this tiling scheme.

        Args:
            min_level: Optional minimum level. Defaults to
                self.min_level.
            max_level: Optional maximum level. Defaults to
                self.max_level.
            unit_name: The spatial unit for the returned resolutions.

        Returns:
            List of spatial resolutions.
        """
        unit_factor = get_unit_factor(
            self.map_unit_name, unit_name or self.map_unit_name
        )
        res_l0 = unit_factor * self.level_zero_resolution

        min_level = min_level if min_level is not None else self.min_level
        if min_level is None:
            min_level = 0
        max_level = max_level if max_level is not None else self.max_level
        if max_level is None:
            raise ValueError("max_value must be given")

        return [res_l0 / (1 << level) for level in range(min_level, max_level + 1)]

    def get_levels_for_resolutions(
        self, resolutions: Sequence[float], unit_name: str
    ) -> tuple[int, int]:
        """Get the minimum and maximum level indices into this tiling scheme
        for the given spatial *resolutions*.

        The resolutions are typically those of a multi-resolution dataset,
        where the first entry represents level zero, the highest resolution,
        hence the smallest resolution value. Subsequent resolution values
        are monotonically increasing.

        Args:
            resolutions: A sequence of spatial resolutions.
            unit_name: The spatial units for *resolutions*.

        Returns:
            the range of levels in this tiling scheme.
        """
        assert_given(resolutions, name="resolutions")
        assert_instance(unit_name, str, name="unit_name")
        if not isinstance(resolutions, np.ndarray):
            resolutions = np.array(resolutions)

        f_to_map = get_unit_factor(unit_name, self.map_unit_name)
        map_resolutions = f_to_map * resolutions
        map_levels = np.ceil(np.log2(self.level_zero_resolution / map_resolutions))
        min_level = max(int(np.min(map_levels)), 0)
        max_level = max(int(np.max(map_levels)), min_level)

        return min_level, max_level

    def get_resolutions_level(
        self, level: int, resolutions: Sequence[float], unit_name: str
    ) -> int:
        """Get the level in the sequence of spatial *resolutions*
        for given *map_level*.

        The resolutions are typically those of a multi-resolution dataset,
        where the first entry represents level zero, the highest resolution,
        hence the smallest resolution value. Subsequent resolution values
        are monotonically increasing.

        Args:
            level: A level within this tiling scheme.
            resolutions: A sequence of spatial resolutions. Values must
                be monotonically increasing. First entry is the highest
                resolution at level zero.
            unit_name: The spatial units for *resolutions*.

        Returns:
            The multi-resolution level.
        """
        assert_given(resolutions, name="resolutions")
        assert_instance(unit_name, str, name="unit_name")

        f_from_map = get_unit_factor(self.map_unit_name, unit_name)
        # Tile pixel size in dataset units for map tile at level 0
        ds_pix_size_l0 = f_from_map * self.level_zero_resolution
        # Tile pixel size in dataset units for map tile at level
        ds_pix_size = ds_pix_size_l0 / (1 << level)

        num_ds_levels = len(resolutions)

        ds_pix_size_min = resolutions[0]
        if ds_pix_size <= ds_pix_size_min:
            return 0

        ds_pix_size_max = resolutions[-1]
        if ds_pix_size >= ds_pix_size_max:
            return num_ds_levels - 1

        for ds_level in range(num_ds_levels - 1):
            ds_pix_size_1 = resolutions[ds_level]
            ds_pix_size_2 = resolutions[ds_level + 1]
            if ds_pix_size_1 <= ds_pix_size <= ds_pix_size_2:
                r = (ds_pix_size - ds_pix_size_1) / (ds_pix_size_2 - ds_pix_size_1)
                if r < 0.5:
                    return ds_level
                else:
                    return ds_level + 1

        raise RuntimeError("should not come here")

    def get_tile_extent(
        self, tile_x: int, tile_y: int, tile_z: int
    ) -> Optional[tuple[float, float, float, float]]:
        """Get the extent in units of the CRS for the given tile coordinates.

        Args:
            tile_x: The tile's column index
            tile_y: The tile's row index
            tile_z: The tile's level index

        Returns:
            The tile's extent
        """
        if tile_z < 0:
            return None

        zoom_factor = 1 << tile_z

        num_tiles_x0, num_tiles_y0 = self.num_level_zero_tiles

        num_tiles_x = num_tiles_x0 * zoom_factor
        if tile_x < 0 or tile_x >= num_tiles_x:
            return None

        num_tiles_y = num_tiles_y0 * zoom_factor
        if tile_y < 0 or tile_y >= num_tiles_y:
            return None

        map_width = self.map_width
        map_height = self.map_height

        map_x0 = -map_width / 2
        map_y0 = map_height / 2

        map_tile_width = map_width / zoom_factor / num_tiles_x0
        map_tile_height = map_height / zoom_factor / num_tiles_y0

        x1 = map_x0 + tile_x * map_tile_width
        y1 = map_y0 - (tile_y + 1) * map_tile_height

        x2 = map_x0 + (tile_x + 1) * map_tile_width
        y2 = map_y0 - tile_y * map_tile_height

        return x1, y1, x2, y2


TilingScheme.WEB_MERCATOR = TilingScheme(
    num_level_zero_tiles=(1, 1),
    crs_name=WEB_MERCATOR_CRS_NAME,
    map_height=EARTH_CIRCUMFERENCE_WGS84,
)

TilingScheme.GEOGRAPHIC = TilingScheme(
    num_level_zero_tiles=(2, 1), crs_name=GEOGRAPHIC_CRS_NAME, map_height=180.0
)


def get_unit_factor(unit_name_from: str, unit_name_to: str) -> float:
    """Get the factor to convert from one unit into another
    with units given by *unit_name_from* and *unit_name_to*.
    """
    from_meter = _is_meter_unit(unit_name_from)
    from_degree = _is_degree_unit(unit_name_from)
    if not from_meter and not from_degree:
        raise ValueError(
            f"unsupported units {unit_name_from!r}."
            f" Unit must be either meters or degrees."
        )

    to_meter = _is_meter_unit(unit_name_to)
    to_degree = _is_degree_unit(unit_name_to)
    if not to_meter and not to_degree:
        raise ValueError(
            f"unsupported units {unit_name_from!r}."
            f" Unit must be either meters or degrees."
        )

    if from_meter and to_degree:
        return 360 / EARTH_CIRCUMFERENCE_WGS84
    if from_degree and to_meter:
        return EARTH_CIRCUMFERENCE_WGS84 / 360
    return 1.0


def subdivide_size(
    size: tuple[int, int], tile_size: tuple[int, int]
) -> list[tuple[int, int]]:
    x_size, y_size = size
    tile_size_x, tile_size_y = tile_size
    sizes = [(x_size, y_size)]
    while True:
        if x_size <= tile_size_x or y_size <= tile_size_y:
            break
        x_size = (x_size + 1) // 2
        y_size = (y_size + 1) // 2
        sizes.append((x_size, y_size))
    return sizes


def get_num_levels(size: tuple[int, int], tile_size: tuple[int, int]) -> int:
    return len(subdivide_size(size, tile_size))


def _is_meter_unit(unit_name: str) -> bool:
    return unit_name.lower() in ("m", "metre", "metres", "meter", "meters")


def _is_degree_unit(unit_name: str) -> bool:
    return unit_name.lower() in (
        "°",
        "deg",
        "degree",
        "degrees",
        "decimal_degree",
        "decimal_degrees",
    )
