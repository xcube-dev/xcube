# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.


from typing import Dict, Union

import pyproj

CrsLike = Union[str, pyproj.CRS]


class ProjCache:
    """
    A cache for pyproj objects
    that may take considerable time to construct.
    """

    INSTANCE: "ProjCache"

    def __init__(self):
        self._crs_cache: dict[str, pyproj.CRS] = dict()
        self._transformer_cache: dict[str, pyproj.Transformer] = dict()

    def get_crs(self, crs: CrsLike) -> pyproj.CRS:
        if isinstance(crs, pyproj.CRS):
            return crs
        key = crs
        if key not in self._crs_cache:
            # pyproj.CRS.from_string() is expensive
            # save result for later use
            self._crs_cache[key] = pyproj.CRS.from_string(crs)
        return self._crs_cache[key]

    def get_transformer(self, crs1: CrsLike, crs2: CrsLike) -> pyproj.Transformer:
        crs1_key = self.get_crs_srs(crs1)
        crs2_key = self.get_crs_srs(crs2)
        key = f"{crs1_key}->{crs2_key}"
        if key not in self._transformer_cache:
            crs1 = self.get_crs(crs1)
            crs2 = self.get_crs(crs2)
            # pyproj.Transformer.from_crs() is really expensive (~0.5 secs)
            # save result for later use
            self._transformer_cache[key] = pyproj.Transformer.from_crs(
                crs1, crs2, always_xy=True
            )
        return self._transformer_cache[key]

    @classmethod
    def get_crs_srs(cls, crs: CrsLike) -> str:
        if isinstance(crs, pyproj.CRS):
            return crs.srs
        return crs


ProjCache.INSTANCE = ProjCache()
