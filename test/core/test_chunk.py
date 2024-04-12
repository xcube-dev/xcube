# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import collections.abc
import unittest

import numpy as np

from test.sampledata import new_test_dataset
from xcube.core.chunk import chunk_dataset
from xcube.core.chunk import compute_chunk_slices
from xcube.core.chunk import get_empty_dataset_chunks
from xcube.core.new import new_cube


class ChunkDatasetTest(unittest.TestCase):
    def test_chunk_dataset(self):
        dataset = new_test_dataset(
            ["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-04", "2010-01-05"],
            precipitation=0.4,
            temperature=275.2,
        )

        chunked_dataset = chunk_dataset(
            dataset, chunk_sizes=dict(time=1, lat=10, lon=20), format_name="zarr"
        )
        self.assertEqual(
            {"chunks": (1, 10, 20)}, chunked_dataset.precipitation.encoding
        )
        self.assertEqual({"chunks": (1, 10, 20)}, chunked_dataset.temperature.encoding)

        chunked_dataset = chunk_dataset(
            dataset, chunk_sizes=dict(time=1, lat=20, lon=40), format_name="netcdf4"
        )
        self.assertEqual(
            {"chunksizes": (1, 20, 40)}, chunked_dataset.precipitation.encoding
        )
        self.assertEqual(
            {"chunksizes": (1, 20, 40)}, chunked_dataset.temperature.encoding
        )

        chunked_dataset = chunk_dataset(
            dataset, chunk_sizes=dict(time=1, lat=20, lon=40)
        )
        self.assertEqual({}, chunked_dataset.precipitation.encoding)
        self.assertEqual({}, chunked_dataset.temperature.encoding)

        dataset = dataset.chunk(dict(time=2, lat=10, lon=20))

        chunked_dataset = chunk_dataset(dataset, chunk_sizes=None, format_name="zarr")
        self.assertEqual({}, chunked_dataset.precipitation.encoding)
        self.assertEqual({}, chunked_dataset.temperature.encoding)

        chunked_dataset = chunk_dataset(dataset, chunk_sizes={}, format_name="zarr")
        self.assertEqual(
            {"chunks": (2, 10, 20)}, chunked_dataset.precipitation.encoding
        )
        self.assertEqual({"chunks": (2, 10, 20)}, chunked_dataset.temperature.encoding)

        chunked_dataset = chunk_dataset(
            dataset, chunk_sizes=dict(time=1), format_name="zarr"
        )
        self.assertEqual(
            {"chunks": (1, 10, 20)}, chunked_dataset.precipitation.encoding
        )
        self.assertEqual({"chunks": (1, 10, 20)}, chunked_dataset.temperature.encoding)

    def test_chunk_dataset_data_vars_only(self):
        cube = chunk_dataset(
            new_cube(
                time_periods=5,
                time_freq="1D",
                time_start="2019-01-01",
                variables=dict(precipitation=0.1, temperature=270.5, soil_moisture=0.2),
            ),
            dict(time=1, lat=90, lon=90),
            format_name="zarr",
            data_vars_only=True,
        )
        self.assertEqual((1, 90, 90), cube.precipitation.data.chunksize)
        self.assertEqual({"chunks": (1, 90, 90)}, cube.precipitation.encoding)
        self.assertEqual((1, 90, 90), cube.precipitation.data.chunksize)
        self.assertEqual({"chunks": (1, 90, 90)}, cube.temperature.encoding)
        self.assertEqual((1, 90, 90), cube.precipitation.data.chunksize)
        self.assertEqual({"chunks": (1, 90, 90)}, cube.soil_moisture.encoding)
        self.assertIsNone(cube.lat.chunks)
        self.assertEqual({}, cube.lat.encoding)
        self.assertIsNone(cube.lon.chunks)
        self.assertEqual({}, cube.lon.encoding)
        self.assertIsNone(cube.lat_bnds.chunks)
        self.assertEqual({}, cube.lat_bnds.encoding)
        self.assertIsNone(cube.lon_bnds.chunks)
        self.assertEqual({}, cube.lon_bnds.encoding)
        self.assertIsNone(cube.time.chunks)
        self.assertNotIn("chunks", cube.time.encoding)

    def test_unchunk_dataset(self):
        dataset = new_test_dataset(
            ["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-04", "2010-01-05"],
            precipitation=0.4,
            temperature=275.2,
        )

        for var in dataset.data_vars.values():
            var.encoding.update({"chunks": (5, 180, 360), "_FillValue": -999.0})

        chunked_dataset = chunk_dataset(dataset, format_name="zarr")
        self.assertEqual({"_FillValue": -999.0}, chunked_dataset.precipitation.encoding)
        self.assertEqual({"_FillValue": -999.0}, chunked_dataset.temperature.encoding)


class GetEmptyDatasetChunksTest(unittest.TestCase):
    def test_not_chunked(self):
        dataset = new_test_dataset(
            ["2010-01-01", "2010-01-02"], precipitation=0.4, temperature=275.2
        )
        empty_dataset_chunks = get_empty_dataset_chunks(dataset)
        self.assertIsInstance(empty_dataset_chunks, collections.abc.Iterator)
        self.assertFalse(isinstance(empty_dataset_chunks, (list, tuple)))
        self.assertEqual(
            [("precipitation", ()), ("temperature", ())],
            [(v, tuple(c)) for v, c in empty_dataset_chunks],
        )

    def test_non_empty(self):
        dataset = new_test_dataset(
            ["2010-01-01", "2010-01-02"], precipitation=0.4, temperature=275.2
        ).chunk(dict(time=1, lat=90, lon=90))
        empty_dataset_chunks = get_empty_dataset_chunks(dataset)
        self.assertIsInstance(empty_dataset_chunks, collections.abc.Iterator)
        self.assertFalse(isinstance(empty_dataset_chunks, (list, tuple)))
        self.assertEqual(
            [("precipitation", ()), ("temperature", ())],
            [(v, tuple(c)) for v, c in empty_dataset_chunks],
        )

    def test_all_empty(self):
        dataset = new_test_dataset(
            ["2010-01-01", "2010-01-02"], precipitation=np.nan, temperature=np.nan
        ).chunk(dict(time=1, lat=90, lon=90))
        empty_dataset_chunks = get_empty_dataset_chunks(dataset)
        self.assertIsInstance(empty_dataset_chunks, collections.abc.Iterator)
        self.assertFalse(isinstance(empty_dataset_chunks, (list, tuple)))
        self.assertEqual(
            [
                (
                    "precipitation",
                    (
                        (0, 0, 0),
                        (0, 0, 1),
                        (0, 0, 2),
                        (0, 0, 3),
                        (0, 1, 0),
                        (0, 1, 1),
                        (0, 1, 2),
                        (0, 1, 3),
                        (1, 0, 0),
                        (1, 0, 1),
                        (1, 0, 2),
                        (1, 0, 3),
                        (1, 1, 0),
                        (1, 1, 1),
                        (1, 1, 2),
                        (1, 1, 3),
                    ),
                ),
                (
                    "temperature",
                    (
                        (0, 0, 0),
                        (0, 0, 1),
                        (0, 0, 2),
                        (0, 0, 3),
                        (0, 1, 0),
                        (0, 1, 1),
                        (0, 1, 2),
                        (0, 1, 3),
                        (1, 0, 0),
                        (1, 0, 1),
                        (1, 0, 2),
                        (1, 0, 3),
                        (1, 1, 0),
                        (1, 1, 1),
                        (1, 1, 2),
                        (1, 1, 3),
                    ),
                ),
            ],
            [(v, tuple(c)) for v, c in empty_dataset_chunks],
        )


class ComputeChunkSlicesTest(unittest.TestCase):
    def test_compute_chunk_slices(self):
        chunk_slices = compute_chunk_slices(((1, 1), (90, 90), (90, 90, 90, 90)))
        self.assertEqual(
            [
                ((0, 0, 0), ((0, 1), (0, 90), (0, 90))),
                ((0, 0, 1), ((0, 1), (0, 90), (90, 180))),
                ((0, 0, 2), ((0, 1), (0, 90), (180, 270))),
                ((0, 0, 3), ((0, 1), (0, 90), (270, 360))),
                ((0, 1, 0), ((0, 1), (90, 180), (0, 90))),
                ((0, 1, 1), ((0, 1), (90, 180), (90, 180))),
                ((0, 1, 2), ((0, 1), (90, 180), (180, 270))),
                ((0, 1, 3), ((0, 1), (90, 180), (270, 360))),
                ((1, 0, 0), ((1, 2), (0, 90), (0, 90))),
                ((1, 0, 1), ((1, 2), (0, 90), (90, 180))),
                ((1, 0, 2), ((1, 2), (0, 90), (180, 270))),
                ((1, 0, 3), ((1, 2), (0, 90), (270, 360))),
                ((1, 1, 0), ((1, 2), (90, 180), (0, 90))),
                ((1, 1, 1), ((1, 2), (90, 180), (90, 180))),
                ((1, 1, 2), ((1, 2), (90, 180), (180, 270))),
                ((1, 1, 3), ((1, 2), (90, 180), (270, 360))),
            ],
            list(chunk_slices),
        )
