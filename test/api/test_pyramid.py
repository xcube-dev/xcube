import os
import time
import unittest

import numpy as np
import xarray as xr

from xcube.api.pyramid import compute_pyramid_levels, write_pyramid_levels, read_pyramid_levels
from xcube.util.dsio import rimraf


def get_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)


class PyramidizeTest(unittest.TestCase):
    @classmethod
    def create_test_dataset(cls, shape=(5, 200, 400), chunks=(1, 25, 25)):
        size = int(np.prod(shape))
        dims = ["time", "y", "x"]
        a_data = np.linspace(0, 1, size, dtype=np.float64).reshape(shape)
        a = xr.DataArray(a_data, dims=dims)
        a.encoding.update(chunks=chunks)
        b_data = np.linspace(-1, 0, size, dtype=np.float64).reshape(shape)
        b = xr.DataArray(b_data, dims=dims)
        b.encoding.update(chunks=chunks)
        return xr.Dataset(dict(a=a, b=b)).chunk(chunks={dims[i]: chunks[i] for i in range(len(dims))})

    def test_compute_pyramid_levels(self):
        dataset = self.create_test_dataset()
        levels = compute_pyramid_levels(dataset)

        self._assert_levels_ok(levels)

    def test_write_read_pyramid_levels(self):
        input_path = get_path("pyramid-input.zarr")
        output_path = get_path("pyramid-output")

        rimraf(input_path)
        rimraf(output_path)

        try:
            dataset = self.create_test_dataset()
            dataset.to_zarr(input_path)

            t0 = time.perf_counter()

            levels = write_pyramid_levels(output_path,
                                          dataset=dataset,
                                          input_path=input_path)

            print(f"write time total: ", time.perf_counter() - t0)

            self._assert_levels_ok(levels)

            t0 = time.perf_counter()

            levels = read_pyramid_levels(output_path)

            print(f"read time total: ", time.perf_counter() - t0)

            self._assert_levels_ok(levels)

        finally:
            rimraf(input_path)
            rimraf(output_path)

    def _assert_levels_ok(self, levels):
        self.assertIsInstance(levels, list)
        self.assertEqual(4, len(levels))

        for i in range(4):
            msg = f"at index {i}"
            self.assertIsInstance(levels[i], xr.Dataset, msg=msg)
            self.assertEqual(2, len(levels[i].data_vars), msg=msg)
            self.assertIn("a", levels[i], msg=msg)
            self.assertIn("b", levels[i], msg=msg)

        expected_shapes = [
            (5, 200, 400),
            (5, 100, 200),
            (5, 50, 100),
            (5, 25, 50),
        ]
        for i in range(4):
            msg = f"at index {i}"
            self.assertEqual(expected_shapes[i], levels[i]["a"].shape, msg=msg)
            self.assertEqual(expected_shapes[i], levels[i]["b"].shape, msg=msg)

        expected_chunks = [
            ((1,) * 5, (25,) * 8, (25,) * 16),
            ((1,) * 5, (25,) * 4, (25,) * 8),
            ((1,) * 5, (25,) * 2, (25,) * 4),
            ((1,) * 5, (25,) * 1, (25,) * 2),
        ]
        for i in range(4):
            msg = f"at index {i}"
            self.assertEqual(expected_chunks[i], levels[i]["a"].chunks, msg=msg)
            self.assertEqual(expected_chunks[i], levels[i]["b"].chunks, msg=msg)
