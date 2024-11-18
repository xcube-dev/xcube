# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.
from typing import Optional

import pyproj
import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.server.api import ApiError
from xcube.webapi.ows.coverages.util import get_h_dim, get_v_dim
from xcube.webapi.ows.coverages.request import CoverageRequest


class CoverageScaling:
    """Representation of a scaling applied to an OGC coverage

    This class represents the scaling specified in an OGC coverage request.
    It is instantiated using a `CoverageRequest` instance, and can apply
    itself to a `GridMapping` instance.
    """

    _scale: Optional[tuple[float, float]] = None
    _final_size: Optional[tuple[int, int]] = None
    _initial_size: tuple[int, int]
    _crs: pyproj.CRS
    _x_name: str
    _y_name: str

    def __init__(self, request: CoverageRequest, crs: pyproj.CRS, ds: xr.Dataset):
        """Create a new scaling from a coverages request object

        Args:
            request: a request optionally including scaling parameters.
                If any scaling parameters are present, the returned
                instance will correspond to them. If no scaling
                parameters are present, a default scaling will be used
                (currently 1:1)
            crs: the CRS of the dataset to be scaled
            ds: the dataset to be scaled
        """
        h_dim = get_h_dim(ds)
        v_dim = get_v_dim(ds)
        for d in h_dim, v_dim:
            size = ds.sizes[d]
            if size == 0:
                # Requirement 8C currently specifies a 204 rather than 404 here,
                # but spec will soon be updated to allow 404 as an alternative.
                # (J. Jacovella-St-Louis, pers. comm., 2023-11-27).
                raise ApiError.NotFound(
                    f"Requested coverage contains no data: {d} has zero size."
                )
        self._initial_size = ds.sizes[h_dim], ds.sizes[v_dim]

        self._crs = crs
        self._y_name = self.get_axis_from_crs(
            {"lat", "latitude", "geodetic latitude", "n", "north", "y"}
        )
        self._x_name = self.get_axis_from_crs(
            {"longitude", "geodetic longitude", "lon", "e", "east", "x"}
        )

        # The standard doesn't define behaviour when multiple scale
        # parameters are given. We choose to handle one and ignore the
        # others in such cases.
        if request.scale_factor is not None:
            self._scale = request.scale_factor, request.scale_factor
        elif request.scale_axes is not None:
            self._scale = self._get_xy_values(request.scale_axes)
        elif request.scale_size is not None:
            # The standard allows floats for "scale-size" but mandates:
            # "The returned coverage SHALL contain exactly the specified number
            # of sample values". We can't return a fractional number of sample
            # values, so truncate to int here.
            x, y = self._get_xy_values(request.scale_size)
            self._final_size = int(x), int(y)
        else:
            # The standard doesn't mandate this as a default; in future, we
            # might choose to downscale automatically if a large coverage
            # is requested without an explicit scaling parameter.
            self._scale = (1, 1)

    @property
    def factor(self) -> tuple[float, float]:
        """Return the two-dimensional scale factor of this scaling

        The components of the scale tuple are expressed as downscaling
        factors: values greater than 1 imply that the rescaled size
        of the coverage in the corresponding dimension will be smaller than
        the original size, and vice versa.

        If the scaling was initially specified as a final size rather than
        a factor, the factor property is an estimate based on the dataset
        dimensions; the effective factor may be different when the scaling
        is applied to a GridMapping.

        Returns:
            a 2-tuple of the x and y scale factors, in that order
        """
        if self._scale is not None:
            return self._scale
        else:
            x_initial, y_initial = self._initial_size
            x_final, y_final = self._final_size
            return x_initial / x_final, y_initial / y_final

    @property
    def size(self) -> tuple[float, float]:
        """Return the final coverage size produced by this scaling

        Returns:
            a 2-tuple of the scaled x and y sizes, in that order, in
            pixels
        """
        if self._final_size is not None:
            return self._final_size
        else:
            x_initial, y_initial = self._initial_size
            x_scale, y_scale = self._scale
            return x_initial / x_scale, y_initial / y_scale

    def _get_xy_values(self, axis_to_value: dict[str, float]) -> tuple[float, float]:
        x, y = None, None
        for axis in axis_to_value:
            if axis.lower()[:3] in ["x", "e", "eas", "lon", self._x_name]:
                x = axis_to_value[axis]
            if axis.lower()[:3] in ["y", "n", "nor", "lat", self._y_name]:
                y = axis_to_value[axis]
        return x, y

    def get_axis_from_crs(self, valid_identifiers: set[str]) -> Optional[str]:
        """Find an axis abbreviation via a set of possible axis identifiers

        This method operates on the CRS with which this scaling was
        instantiated. It returns the abbreviation of the first axis in
        the CRS which has either a name or abbreviation matching any string
        in the supplied set.

        Args:
            valid_identifiers: a set of axis identifiers

        Returns:
            the abbreviation of the first axis in this scaling's CRS
            whose name or abbreviation is in the supplied set, or `None`
            if no such axis exists
        """
        for axis in self._crs.axis_info:
            if not hasattr(axis, "abbrev"):
                continue
            identifiers = set(
                map(
                    lambda attr: getattr(axis, attr, "").lower(),
                    ["name", "abbrev"],
                )
            )
            if identifiers & valid_identifiers:
                return axis.abbrev
        return None

    def apply(self, gm: GridMapping) -> GridMapping:
        """Apply this scaling to a grid mapping

        The supplied grid mapping is regularized before being scaled.

        Args:
            gm: a grid mapping to be scaled

        Returns:
            the supplied grid mapping, scaled according to this scaling.
            If this scaling is 1:1, the returned grid mapping may be the
            original object.
        """
        if self.factor == (1, 1):
            return gm
        else:
            regular = gm.to_regular()
            source = regular.size
            # Even if the scaling was specified as a factor, we calculate
            # from the (inferred) final size. If a factor was given,
            # self.size is the final size as calculated from the originally
            # specified dataset, which is what the client would expect. The
            # regularized GridMapping might have a different size,
            # so we don't want to apply a specified factor to it directly.
            return regular.scale((self.size[0] / source[0], self.size[1] / source[1]))
