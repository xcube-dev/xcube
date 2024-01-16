import collections.abc
import unittest

import numpy as np

from test.sampledata import new_test_dataset
from xcube.core.chunk import chunk_dataset
from xcube.core.chunk import compute_chunk_slices
from xcube.core.chunk import get_empty_dataset_chunks


class ChunkDatasetTest(unittest.TestCase):

    def test_chunk_dataset(self):
        dataset = new_test_dataset(["2010-01-01", "2010-01-02",
                                    "2010-01-03", "2010-01-04",
                                    "2010-01-05"],
                                   precipitation=0.4, temperature=275.2)

        chunked_dataset = chunk_dataset(dataset,
                                        chunk_sizes=dict(time=1,
                                                         lat=10, lon=20),
                                        format_name="zarr")
        self.assertEqual({'chunks': (1, 10, 20)},
                         chunked_dataset.precipitation.encoding)
        self.assertEqual({'chunks': (1, 10, 20)},
                         chunked_dataset.temperature.encoding)

        chunked_dataset = chunk_dataset(dataset,
                                        chunk_sizes=dict(time=1,
                                                         lat=20, lon=40),
                                        format_name="netcdf4")
        self.assertEqual({'chunksizes': (1, 20, 40)},
                         chunked_dataset.precipitation.encoding)
        self.assertEqual({'chunksizes': (1, 20, 40)},
                         chunked_dataset.temperature.encoding)

        chunked_dataset = chunk_dataset(dataset,
                                        chunk_sizes=dict(time=1,
                                                         lat=20, lon=40))
        self.assertEqual({}, chunked_dataset.precipitation.encoding)
        self.assertEqual({}, chunked_dataset.temperature.encoding)

        dataset = dataset.chunk(dict(time=2, lat=10, lon=20))

        chunked_dataset = chunk_dataset(dataset,
                                        chunk_sizes=None,
                                        format_name="zarr")
        self.assertEqual({}, chunked_dataset.precipitation.encoding)
        self.assertEqual({}, chunked_dataset.temperature.encoding)

        chunked_dataset = chunk_dataset(dataset,
                                        chunk_sizes={},
                                        format_name="zarr")
        self.assertEqual({'chunks': (2, 10, 20)},
                         chunked_dataset.precipitation.encoding)
        self.assertEqual({'chunks': (2, 10, 20)},
                         chunked_dataset.temperature.encoding)

        chunked_dataset = chunk_dataset(dataset,
                                        chunk_sizes=dict(time=1),
                                        format_name="zarr")
        self.assertEqual({'chunks': (1, 10, 20)},
                         chunked_dataset.precipitation.encoding)
        self.assertEqual({'chunks': (1, 10, 20)},
                         chunked_dataset.temperature.encoding)

    def test_unchunk_dataset(self):
        dataset = new_test_dataset(["2010-01-01", "2010-01-02",
                                    "2010-01-03", "2010-01-04",
                                    "2010-01-05"],
                                   precipitation=0.4,
                                   temperature=275.2)

        for var in dataset.data_vars.values():
            var.encoding.update({"chunks": (5, 180, 360),
                                 "_FillValue": -999.0})

        chunked_dataset = chunk_dataset(dataset, format_name="zarr")
        self.assertEqual({"_FillValue": -999.0},
                         chunked_dataset.precipitation.encoding)
        self.assertEqual({"_FillValue": -999.0},
                         chunked_dataset.temperature.encoding)


class GetEmptyDatasetChunksTest(unittest.TestCase):

    def test_not_chunked(self):
        dataset = new_test_dataset(["2010-01-01", "2010-01-02"],
                                   precipitation=0.4,
                                   temperature=275.2)
        empty_dataset_chunks = get_empty_dataset_chunks(dataset)
        self.assertIsInstance(empty_dataset_chunks, collections.abc.Iterator)
        self.assertFalse(isinstance(empty_dataset_chunks, (list, tuple)))
        self.assertEqual([('precipitation', ()),
                          ('temperature', ())],
                         [(v, tuple(c)) for v, c in empty_dataset_chunks])

    def test_non_empty(self):
        dataset = new_test_dataset(["2010-01-01", "2010-01-02"],
                                   precipitation=0.4,
                                   temperature=275.2) \
            .chunk(dict(time=1, lat=90, lon=90))
        empty_dataset_chunks = get_empty_dataset_chunks(dataset)
        self.assertIsInstance(empty_dataset_chunks, collections.abc.Iterator)
        self.assertFalse(isinstance(empty_dataset_chunks, (list, tuple)))
        self.assertEqual([('precipitation', ()),
                          ('temperature', ())],
                         [(v, tuple(c)) for v, c in empty_dataset_chunks])

    def test_all_empty(self):
        dataset = new_test_dataset(["2010-01-01", "2010-01-02"],
                                   precipitation=np.nan,
                                   temperature=np.nan) \
            .chunk(dict(time=1, lat=90, lon=90))
        empty_dataset_chunks = get_empty_dataset_chunks(dataset)
        self.assertIsInstance(empty_dataset_chunks, collections.abc.Iterator)
        self.assertFalse(isinstance(empty_dataset_chunks, (list, tuple)))
        self.assertEqual([('precipitation', ((0, 0, 0),
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
                                             (1, 1, 3))),
                          ('temperature', ((0, 0, 0),
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
                                           (1, 1, 3)))],
                         [(v, tuple(c)) for v, c in empty_dataset_chunks])


class ComputeChunkSlicesTest(unittest.TestCase):

    def test_compute_chunk_slices(self):
        chunk_slices = compute_chunk_slices(((1, 1),
                                             (90, 90), (90, 90, 90, 90)))
        self.assertEqual([((0, 0, 0), ((0, 1), (0, 90), (0, 90))),
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
                          ((1, 1, 3), ((1, 2), (90, 180), (270, 360)))],
                         list(chunk_slices))
