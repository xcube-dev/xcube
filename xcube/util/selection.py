# Copyright (c) 2018-2026 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from collections.abc import Hashable, Mapping
from threading import RLock
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

NearestLookup = tuple[pd.Index, np.ndarray | None]


class NearestIndexCache:
    """Cache nearest-label lookups for the lifetime of an opened dataset.

    Entries retain their source indexes and are keyed by identity, so variables
    sharing an index reuse a lookup, while different pyramid levels or changed
    coordinates cannot accidentally reuse another index's positions.
    """

    def __init__(self):
        self._lookups: dict[int, tuple[pd.Index, NearestLookup]] = {}
        self._lock = RLock()

    def get_lookup(self, index: pd.Index) -> NearestLookup:
        with self._lock:
            key = id(index)
            if key not in self._lookups:
                if index.is_unique and (
                    index.is_monotonic_increasing or index.is_monotonic_decreasing
                ):
                    lookup = index, None
                else:
                    positions = np.flatnonzero(~index.duplicated(keep="first"))
                    positions = positions[index.take(positions).argsort()]
                    lookup = index.take(positions), positions
                self._lookups[key] = index, lookup
            return self._lookups[key][1]


def select_nearest(
    variable: xr.DataArray,
    labels: Mapping[Hashable, Any],
    index_cache: NearestIndexCache | None = None,
) -> xr.DataArray:
    """Select scalar nearest labels, choosing the first duplicate occurrence.

    Numeric and datetime coordinates may be unsorted or non-unique. Ties between
    distinct labels prefer the larger label, as with pandas nearest selection.
    Pass a dataset-scoped cache to reuse lookups across requests and variables.
    """
    if index_cache is None:
        index_cache = NearestIndexCache()
    indexes = variable.indexes
    indexers = {}
    for dim, label in labels.items():
        index, positions = index_cache.get_lookup(indexes[dim])
        if isinstance(label, np.ndarray) and label.ndim == 0:
            label = label[()]
        match = index.get_indexer([label], method="nearest")[0]
        if match < 0:
            raise ValueError(f"No match for coordinate {dim!r}: {label!r}")
        indexers[dim] = int(match if positions is None else positions[match])
    return variable.isel(indexers)
