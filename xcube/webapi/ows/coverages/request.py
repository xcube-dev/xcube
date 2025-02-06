# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.
import re
from collections.abc import Mapping, Sequence
from typing import Optional, Union

import pyproj
import rfc3339_validator


class CoverageRequest:
    """A representation of a parsed OGC API - Coverages request

    As defined in https://docs.ogc.org/DRAFTS/19-087.html
    (§7.2.4–11.2.1)
    """

    properties: Optional[list[str]]
    bbox: Optional[list[float]]
    bbox_crs: pyproj.CRS
    datetime: Optional[Union[str, tuple[Optional[str], Optional[str]]]]
    subset: Optional[dict[str, Union[str, tuple[Optional[str], Optional[str]]]]]
    subset_crs: pyproj.CRS
    scale_factor: Optional[float]
    scale_axes: Optional[dict[str, float]]
    scale_size: Optional[dict[str, float]]
    crs: Optional[pyproj.CRS]

    def __init__(self, query: Mapping[str, Sequence[str]]):
        self.query = query
        self._parse_bbox()
        self.bbox_crs = self._parse_crs("bbox-crs", "OGC:CRS84")
        self._parse_properties()
        self._parse_datetime()
        self._parse_subset()
        self.subset_crs = self._parse_crs("subset-crs", "OGC:CRS84")
        self._parse_scale_factor()
        self._parse_scale_axes()
        self._parse_scale_size()
        self.crs = self._parse_crs("crs")

    def _parse_bbox(self):
        self.bbox = None
        if "bbox" in self.query:
            bbox_spec = self.query["bbox"][0]
            try:
                self.bbox = list(map(float, bbox_spec.split(",")))
            except ValueError:
                raise ValueError(f'Invalid bbox "{bbox_spec}"')
            if len(self.bbox) != 4:
                # TODO: Handle 3D bounding boxes
                raise ValueError(f'Invalid bbox "{bbox_spec}": must have 4 elements')

    def _parse_datetime(self):
        if "datetime" in self.query:
            datetime_spec = self.query["datetime"][0]
            limits = datetime_spec.split("/")
            if len(limits) > 2:
                raise ValueError(f"Too many parts in datetime {datetime_spec}")
            limits = tuple(None if lt == ".." else lt for lt in limits)
            for limit in limits:
                if limit is not None and not rfc3339_validator.validate_rfc3339(limit):
                    raise ValueError(f"Invalid datetime {limit}")
            self.datetime = limits if len(limits) == 2 else limits[0]
        else:
            self.datetime = None

    def _parse_subset(self):
        if "subset" in self.query:
            self.subset = {}
            subset_spec = self.query["subset"][0]
            for part in subset_spec.split(","):
                # First try matching with quotation marks
                m = re.match('^(.*)[(]"([^")]*)"(?::"(.*)")?[)]$', part)
                if m is None:
                    # If that fails, try without quotation marks
                    m = re.match("^(.*)[(]([^:)]*)(?::(.*))?[)]$", part)
                if m is None:
                    raise ValueError(f'Unrecognized subset specifier "{part}"')
                else:
                    axis, low, high = m.groups()
                if high is None:
                    self.subset[axis] = low
                else:
                    low = None if low == "*" else low
                    high = None if high == "*" else high
                    self.subset[axis] = low, high
        else:
            self.subset = None

    def _parse_properties(self):
        # https://docs.ogc.org/DRAFTS/19-087.html#_parameter_properties
        # TODO: Support * wildcard (Requirement 13E)
        if "properties" in self.query:
            self.properties = self.query["properties"][0].split(",")
        else:
            self.properties = None

    def _parse_scale_factor(self):
        if "scale-factor" in self.query:
            scale_factor_str = self.query["scale-factor"][0]
            try:
                self.scale_factor = float(scale_factor_str)
            except ValueError:
                raise ValueError(f"Invalid scale-factor {scale_factor_str}")
        else:
            # We don't default to 1, since (per the standard) an implementation
            # may choose to downscale by default.
            self.scale_factor = None

    def _parse_scale_axes(self):
        self.scale_axes = (
            self._parse_scale_specifier("scale-axes")
            if "scale-axes" in self.query
            else None
        )

    def _parse_scale_size(self):
        self.scale_size = (
            self._parse_scale_specifier("scale-size")
            if "scale-size" in self.query
            else None
        )

    def _parse_scale_specifier(self, param_name: str) -> dict[str, float]:
        spec = self.query[param_name][0]
        result = {}
        for part in spec.split(","):
            m = re.match("^(.*)[(]([0-9.]+)[)]$", part)
            if m is None:
                raise ValueError(f"Invalid {param_name} value {part}")
            else:
                result[m.group(1)] = float(m.group(2))
        return result

    def _parse_crs(self, param, default=None):
        specifier = self.query[param][0] if param in self.query else default
        if specifier is None:
            return None
        if m := re.match(r"^\[(.*)]$", specifier):
            specifier = m.group(1)
        try:
            return pyproj.CRS(specifier)
        except pyproj.exceptions.CRSError as e:
            raise ValueError(e)
