import os
import time
import unittest

import numpy as np
import xarray as xr

from xcube.core.dsio import rimraf
from xcube.core.level import compute_levels
from xcube.core.level import read_levels
from xcube.core.level import write_levels


def get_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)


class PyramidTest(unittest.TestCase):
    @classmethod
    def create_test_dataset(cls, shape, chunks=None):
        size = int(np.prod(shape))
        dims = ["time", "y", "x"]
        a_data = np.linspace(0, 1, size, dtype=np.float64).reshape(shape)
        a = xr.DataArray(a_data, dims=dims)
        b_data = np.linspace(-1, 0, size, dtype=np.float64).reshape(shape)
        b = xr.DataArray(b_data, dims=dims)
        if chunks:
            a.encoding.update(chunks=chunks, chunksizes=chunks)
            b.encoding.update(chunks=chunks, chunksizes=chunks)
            return xr.Dataset(dict(a=a, b=b)) \
                .chunk(chunks={dims[i]: chunks[i]
                               for i in range(len(dims))})
        else:
            return xr.Dataset(dict(a=a, b=b))

    def test_compute_levels(self):
        dataset = self.create_test_dataset(shape=(5, 200, 400),
                                           chunks=(1, 25, 25))
        levels = compute_levels(dataset)
        self._assert_levels_ok(levels,
                               expected_num_levels=4,
                               expected_shapes=[
                                   (5, 200, 400),
                                   (5, 100, 200),
                                   (5, 50, 100),
                                   (5, 25, 50),
                               ],
                               expected_chunks=[
                                   ((1,) * 5, (25,) * 8, (25,) * 16),
                                   ((1,) * 5, (25,) * 4, (25,) * 8),
                                   ((1,) * 5, (25,) * 2, (25,) * 4),
                                   ((1,) * 5, (25,) * 1, (25,) * 2),
                               ])

    def test_compute_levels_with_max_levels(self):
        dataset = self.create_test_dataset(shape=(5, 200, 400),
                                           chunks=(1, 25, 25))
        levels = compute_levels(dataset, num_levels_max=3)
        self._assert_levels_ok(levels,
                               expected_num_levels=3,
                               expected_shapes=[
                                   (5, 200, 400),
                                   (5, 100, 200),
                                   (5, 50, 100),
                               ],
                               expected_chunks=[
                                   ((1,) * 5, (25,) * 8, (25,) * 16),
                                   ((1,) * 5, (25,) * 4, (25,) * 8),
                                   ((1,) * 5, (25,) * 2, (25,) * 4),
                               ])

    def test_write_read_levels_with_even_sizes(self):
        shape = (5, 200, 400)
        tile_shape = (25, 25)
        expected_num_levels = 4
        expected_shapes = [
            (5, 200, 400),
            (5, 100, 200),
            (5, 50, 100),
            (5, 25, 50),
        ]
        expected_chunks = [
            ((1,) * 5, (25,) * 8, (25,) * 16),
            ((1,) * 5, (25,) * 4, (25,) * 8),
            ((1,) * 5, (25,) * 2, (25,) * 4),
            ((1,) * 5, (25,) * 1, (25,) * 2),
        ]
        self._assert_io_ok(shape,
                           tile_shape,
                           False,
                           expected_num_levels,
                           expected_shapes,
                           expected_chunks)
        self._assert_io_ok(shape,
                           tile_shape,
                           True,
                           expected_num_levels,
                           expected_shapes,
                           expected_chunks)

    def test_write_read_levels_with_odd_sizes(self):
        shape = (5, 203, 405)
        tile_shape = (37, 38)
        expected_num_levels = 3
        expected_shapes = [
            (5, 203, 405),
            (5, 102, 203),
            (5, 51, 102),
        ]
        expected_chunks = [
            ((1,) * 5,
             (37, 37, 37, 37, 37, 18),
             (38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 25)),
            ((1,) * 5,
             (37, 37, 28),
             (38, 38, 38, 38, 38, 13)),
            ((1,) * 5,
             (37, 14),
             (38, 38, 26)),
        ]
        self._assert_io_ok(shape,
                           tile_shape,
                           False,
                           expected_num_levels,
                           expected_shapes,
                           expected_chunks)
        self._assert_io_ok(shape,
                           tile_shape,
                           True,
                           expected_num_levels,
                           expected_shapes,
                           expected_chunks)

    def _assert_io_ok(self,
                      shape,
                      tile_shape,
                      link_input: bool,
                      expected_num_levels,
                      expected_shapes,
                      expected_chunks):

        input_path = get_path("pyramid-input.zarr")
        output_path = get_path("pyramid-output")

        rimraf(input_path)
        rimraf(output_path)

        try:
            dataset = self.create_test_dataset(shape,
                                               chunks=(1, *tile_shape))
            dataset.to_zarr(input_path)

            t0 = time.perf_counter()

            levels = write_levels(output_path,
                                  dataset=dataset,
                                  spatial_tile_shape=tile_shape,
                                  input_path=input_path,
                                  link_input=link_input)

            print(f"write time total: ", time.perf_counter() - t0)

            self._assert_levels_ok(levels,
                                   expected_num_levels,
                                   expected_shapes,
                                   expected_chunks)

            t0 = time.perf_counter()

            levels = read_levels(output_path)

            print(f"read time total: ", time.perf_counter() - t0)

            self._assert_levels_ok(levels,
                                   expected_num_levels,
                                   expected_shapes,
                                   expected_chunks)

        finally:
            rimraf(input_path)
            rimraf(output_path)

    def _assert_levels_ok(self,
                          levels,
                          expected_num_levels,
                          expected_shapes,
                          expected_chunks):
        self.assertIsInstance(levels, list)
        self.assertEqual(expected_num_levels, len(levels))

        for i in range(expected_num_levels):
            msg = f"at index {i}"
            self.assertIsInstance(levels[i], xr.Dataset, msg=msg)
            self.assertEqual(2, len(levels[i].data_vars), msg=msg)
            self.assertIn("a", levels[i], msg=msg)
            self.assertIn("b", levels[i], msg=msg)

        for i in range(expected_num_levels):
            msg = f"at index {i}"
            self.assertEqual(expected_shapes[i], levels[i]["a"].shape,
                             msg=msg)
            self.assertEqual(expected_shapes[i], levels[i]["b"].shape,
                             msg=msg)

        for i in range(expected_num_levels):
            msg = f"at index {i}"
            self.assertEqual(expected_chunks[i], levels[i]["a"].chunks,
                             msg=msg)
            self.assertEqual(expected_chunks[i], levels[i]["b"].chunks,
                             msg=msg)
