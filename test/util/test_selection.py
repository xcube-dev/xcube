# Copyright (c) 2018-2026 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from xcube.util.selection import NearestIndexCache, select_nearest


class SelectNearestTest(unittest.TestCase):
    def test_duplicate_unsorted_coordinates(self):
        var = xr.DataArray(
            [20, 10, 99, 30], dims="depth", coords={"depth": [2, 1, 1, 3]}
        )
        cache = NearestIndexCache()
        for label, expected in [(1, 10), (1.1, 10), (1.5, 20), (-1, 10), (4, 30)]:
            with self.subTest(label=label):
                actual = select_nearest(var, {"depth": label}, cache)
                self.assertEqual(expected, actual.item())
                self.assertEqual((), actual.dims)
        np.testing.assert_array_equal(var.depth, [2, 1, 1, 3])

    def test_matches_xarray_for_unique_coordinates(self):
        for coords in ([1, 2, 3], [3, 2, 1]):
            var = xr.DataArray([10, 20, 30], dims="depth", coords={"depth": coords})
            for label in [0, 1, 1.5, 2.9, 4]:
                with self.subTest(coords=coords, label=label):
                    xr.testing.assert_identical(
                        var.sel(depth=label, method="nearest"),
                        select_nearest(var, {"depth": label}),
                    )

    def test_datetime_and_multiple_dimensions_stay_lazy(self):
        var = xr.DataArray(
            np.arange(12).reshape(3, 4),
            dims=("time", "depth"),
            coords={
                "time": pd.to_datetime(["2026-09-03", "2026-09-01", "2026-09-01"]),
                "depth": [2, 1, 1, 3],
            },
        ).chunk()
        actual = select_nearest(
            var,
            {"time": np.array("2026-09-01T06:00", dtype="datetime64[m]"), "depth": 1.1},
        )
        self.assertIsNotNone(actual.chunks)
        xr.testing.assert_identical(var.isel(time=1, depth=1), actual)

    def test_empty_coordinate_has_no_match(self):
        var = xr.DataArray([], dims="depth", coords={"depth": []})
        with self.assertRaisesRegex(ValueError, "No match for coordinate 'depth'"):
            select_nearest(var, {"depth": 0})

    def test_cache_reuses_lookup_across_variables_and_threads(self):
        ds = xr.Dataset(
            {"a": ("depth", [10, 20, 30]), "b": ("depth", [40, 50, 60])},
            coords={"depth": [2, 1, 1]},
        )
        cache = NearestIndexCache()
        index = ds.indexes["depth"]
        with patch.object(index, "duplicated", wraps=index.duplicated) as duplicated:
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(
                    pool.map(
                        lambda name: select_nearest(
                            ds[name], {"depth": 1}, cache
                        ).item(),
                        ["a", "b"] * 4,
                    )
                )
            self.assertEqual([20, 50] * 4, results)
            duplicated.assert_called_once()

    def test_cache_distinguishes_changed_coordinates(self):
        cache = NearestIndexCache()
        var = xr.DataArray([10, 20, 30], dims="depth", coords={"depth": [2, 1, 1]})
        changed = var.assign_coords(depth=[1, 2, 2])
        self.assertEqual(20, select_nearest(var, {"depth": 1}, cache).item())
        self.assertEqual(10, select_nearest(changed, {"depth": 1}, cache).item())
        self.assertEqual(20, select_nearest(var, {"depth": 1}, cache).item())
