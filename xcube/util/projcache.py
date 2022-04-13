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


from typing import Dict, Union

import pyproj

CrsLike = Union[str, pyproj.CRS]


class ProjCache:
    """
    A cache for pyproj objects
    that may take considerable time to construct.
    """
    INSTANCE: 'ProjCache'

    def __init__(self):
        self._crs_cache: Dict[str, pyproj.CRS] = dict()
        self._transformer_cache: Dict[str, pyproj.Transformer] = dict()

    def get_crs(self, crs: CrsLike) -> pyproj.CRS:
        if isinstance(crs, pyproj.CRS):
            return crs
        key = crs
        if key not in self._crs_cache:
            # pyproj.CRS.from_string() is expensive
            # save result for later use
            self._crs_cache[key] = pyproj.CRS.from_string(crs)
        return self._crs_cache[key]

    def get_transformer(self,
                        crs1: CrsLike,
                        crs2: CrsLike) -> pyproj.Transformer:
        crs1_key = self.get_crs_srs(crs1)
        crs2_key = self.get_crs_srs(crs2)
        key = f'{crs1_key}->{crs2_key}'
        if key not in self._transformer_cache:
            crs1 = self.get_crs(crs1)
            crs2 = self.get_crs(crs2)
            # pyproj.Transformer.from_crs() is really expensive (~0.5 secs)
            # save result for later use
            self._transformer_cache[key] = pyproj.Transformer.from_crs(
                crs1, crs2, always_xy=True)
        return self._transformer_cache[key]

    @classmethod
    def get_crs_srs(cls, crs: CrsLike) -> str:
        if isinstance(crs, pyproj.CRS):
            return crs.srs
        return crs


ProjCache.INSTANCE = ProjCache()
