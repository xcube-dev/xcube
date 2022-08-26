import unittest
from typing import Dict, Any

import numpy as np
import pytest
import xarray as xr

from xcube.core.arraystore import GenericArrayInfo
from xcube.core.arraystore import GenericArrayStore
from xcube.core.arraystore import dict_to_bytes
from xcube.core.arraystore import ndarray_to_bytes
from xcube.core.arraystore import str_to_bytes


# noinspection PyMethodMayBeStatic
class GenericArrayInfoTest(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual({},
                         GenericArrayInfo())

    def test_finalize_raises(self):
        data = np.linspace(1, 4, 4)

        def get_data(index):
            return data

        with pytest.raises(ValueError, match="missing array name"):
            GenericArrayInfo(dims=["x"],
                             data=data).finalize()

        with pytest.raises(ValueError,
                           match="array 'x':"
                                 " either data or get_data must be defined"):
            GenericArrayInfo(name="x",
                             dims=["x"]).finalize()

        with pytest.raises(ValueError,
                           match="array 'x':"
                                 " data and get_data"
                                 " cannot be defined together"):
            GenericArrayInfo(name="x",
                             dims=["x"],
                             data=data,
                             get_data=get_data).finalize()

        with pytest.raises(TypeError,
                           match="array 'x': get_data must be a callable"):
            # noinspection PyTypeChecker
            GenericArrayInfo(name="x",
                             dims=["x"],
                             get_data=data).finalize()

        with pytest.raises(ValueError,
                           match="array 'x': missing dims"):
            GenericArrayInfo(name="x",
                             data=data).finalize()

        with pytest.raises(ValueError,
                           match="array 'x': missing dtype"):
            GenericArrayInfo(name="x",
                             dims=["x"],
                             shape=[4],
                             get_data=get_data).finalize()

        with pytest.raises(ValueError,
                           match="array 'x': missing shape"):
            GenericArrayInfo(name="x",
                             dims=["x"],
                             dtype=data.dtype,
                             get_data=get_data).finalize()


class GenericArrayStoreTest(unittest.TestCase):
    @staticmethod
    def new_zarr_store(shape, chunks, get_data) -> GenericArrayStore:
        store = GenericArrayStore(
            array_defaults=GenericArrayInfo(
                dims=("time", "y", "x"),
                shape=shape,
                chunks=chunks
            )
        )

        t_size, y_size, x_size = shape
        store.add_array(name="x", dims="x",
                        data=np.linspace(0., 1., x_size))
        store.add_array(name="y", dims="y",
                        data=np.linspace(0., 1., y_size))
        store.add_array(name="time", dims="time",
                        data=np.linspace(1, 365, t_size))

        store.add_array(name="chl",
                        dtype=np.dtype(np.float32).str,
                        get_data=get_data)

        return store

    def setUp(self) -> None:
        self.chunk_shapes = set()
        self.chunk_indexes = set()

    def get_data(self, chunk_index, chunk_info=None, array_info=None):
        it, iy, ix = chunk_index
        st, sy, sx = array_info["shape"]
        nt, ny, nx = array_info["chunks"]
        pt = it * nt
        py = iy * ny
        px = ix * nx
        value = (pt * sy + py) * sx + px
        self.chunk_shapes.add(chunk_info["shape"])
        self.chunk_indexes.add(chunk_index)
        return np.full((nt, ny, nx), value, dtype=np.float32)

    def test_keys(self):
        store = self.new_zarr_store((3, 6, 8), (1, 2, 4), self.get_data)
        self.assertEqual({
            '.zmetadata',
            '.zgroup',
            '.zattrs',
            'x', 'x/.zarray', 'x/.zattrs',
            'x/0',
            'y', 'y/.zarray', 'y/.zattrs',
            'y/0',
            'time', 'time/.zarray', 'time/.zattrs',
            'time/0',
            'chl', 'chl/.zarray', 'chl/.zattrs',
            'chl/0.0.0', 'chl/0.0.1',
            'chl/0.1.0', 'chl/0.1.1',
            'chl/0.2.0', 'chl/0.2.1',
            'chl/1.0.0', 'chl/1.0.1',
            'chl/1.1.0', 'chl/1.1.1',
            'chl/1.2.0', 'chl/1.2.1',
            'chl/2.0.0', 'chl/2.0.1',
            'chl/2.1.0', 'chl/2.1.1',
            'chl/2.2.0', 'chl/2.2.1',
        }, set(store.keys()))

    def test_listdir(self):
        store = self.new_zarr_store((3, 6, 8), (1, 2, 4), self.get_data)
        self.assertEqual([
            '.zmetadata',
            '.zgroup',
            '.zattrs',
            'x',
            'y',
            'time',
            'chl',
        ], store.listdir(''))

        self.assertEqual([
            'time/.zarray',
            'time/.zattrs',
            'time/0',
        ], store.listdir('time'))

        self.assertEqual([
            'chl/.zarray', 'chl/.zattrs',
            'chl/0.0.0', 'chl/0.0.1',
            'chl/0.1.0', 'chl/0.1.1',
            'chl/0.2.0', 'chl/0.2.1',
            'chl/1.0.0', 'chl/1.0.1',
            'chl/1.1.0', 'chl/1.1.1',
            'chl/1.2.0', 'chl/1.2.1',
            'chl/2.0.0', 'chl/2.0.1',
            'chl/2.1.0', 'chl/2.1.1',
            'chl/2.2.0', 'chl/2.2.1',
        ], store.listdir('chl'))

    def test_zarr_store_not_divisible(self):
        shape = 3, 6, 8
        chunks = 1, 2, 5
        store = self.new_zarr_store(shape, chunks,
                                    get_data=self.get_data)

        ds = xr.open_zarr(store)

        self.assertEqual({'x', 'y', 'time'}, set(ds.coords))
        self.assertEqual({'chl'}, set(ds.data_vars))

        self.assertEqual(np.float32, ds.chl.dtype)
        self.assertEqual(shape, ds.chl.shape)
        ds.chl.load()
        self.assertEqual({(1, 2, 3),
                          (1, 2, 5)}, self.chunk_shapes)
        self.assertEqual({(0, 0, 0),
                          (0, 0, 1),
                          (0, 1, 0),
                          (0, 1, 1),
                          (0, 2, 0),
                          (0, 2, 1),
                          (1, 0, 0),
                          (1, 0, 1),
                          (1, 1, 0),
                          (1, 1, 1),
                          (1, 2, 0),
                          (1, 2, 1),
                          (2, 0, 0),
                          (2, 0, 1),
                          (2, 1, 0),
                          (2, 1, 1),
                          (2, 2, 0),
                          (2, 2, 1)}, self.chunk_indexes)
        print(repr(ds.chl.data[0]))
        np.testing.assert_array_equal(
            ds.chl.data[0],
            np.array(
                [[0., 0., 0., 0., 0., 5., 5., 5.],
                 [0., 0., 0., 0., 0., 5., 5., 5.],
                 [16., 16., 16., 16., 16., 21., 21., 21.],
                 [16., 16., 16., 16., 16., 21., 21., 21.],
                 [32., 32., 32., 32., 32., 37., 37., 37.],
                 [32., 32., 32., 32., 32., 37., 37., 37.]],
                dtype=np.float32
            )
        )
        print(repr(ds.chl.data[1]))
        np.testing.assert_array_equal(
            ds.chl.data[1],
            np.array(
                [[48., 48., 48., 48., 48., 53., 53., 53.],
                 [48., 48., 48., 48., 48., 53., 53., 53.],
                 [64., 64., 64., 64., 64., 69., 69., 69.],
                 [64., 64., 64., 64., 64., 69., 69., 69.],
                 [80., 80., 80., 80., 80., 85., 85., 85.],
                 [80., 80., 80., 80., 80., 85., 85., 85.]],
                dtype=np.float32
            )
        )
        print(repr(ds.chl.data[2]))
        np.testing.assert_array_equal(
            ds.chl.data[2],
            np.array(
                [[96., 96., 96., 96., 96., 101., 101., 101.],
                 [96., 96., 96., 96., 96., 101., 101., 101.],
                 [112., 112., 112., 112., 112., 117., 117., 117.],
                 [112., 112., 112., 112., 112., 117., 117., 117.],
                 [128., 128., 128., 128., 128., 133., 133., 133.],
                 [128., 128., 128., 128., 128., 133., 133., 133.]],
                dtype=np.float32
            )
        )

    def test_zarr_store(self):
        shape = 3, 6, 8
        chunks = 1, 2, 4
        store = self.new_zarr_store(shape, chunks,
                                    get_data=self.get_data)

        ds = xr.open_zarr(store)

        self.assertEqual({'x', 'y', 'time'}, set(ds.coords))
        self.assertEqual({'chl'}, set(ds.data_vars))

        self.assertEqual(np.float32, ds.chl.dtype)
        self.assertEqual(shape, ds.chl.shape)
        ds.chl.load()
        self.assertEqual({(1, 2, 4)}, self.chunk_shapes)
        self.assertEqual({(0, 0, 0),
                          (0, 0, 1),
                          (0, 1, 0),
                          (0, 1, 1),
                          (0, 2, 0),
                          (0, 2, 1),
                          (1, 0, 0),
                          (1, 0, 1),
                          (1, 1, 0),
                          (1, 1, 1),
                          (1, 2, 0),
                          (1, 2, 1),
                          (2, 0, 0),
                          (2, 0, 1),
                          (2, 1, 0),
                          (2, 1, 1),
                          (2, 2, 0),
                          (2, 2, 1)}, self.chunk_indexes)
        print(repr(ds.chl.data[0]))
        np.testing.assert_array_equal(
            ds.chl.data[0],
            np.array(
                [[0., 0., 0., 0., 4., 4., 4., 4.],
                 [0., 0., 0., 0., 4., 4., 4., 4.],
                 [16., 16., 16., 16., 20., 20., 20., 20.],
                 [16., 16., 16., 16., 20., 20., 20., 20.],
                 [32., 32., 32., 32., 36., 36., 36., 36.],
                 [32., 32., 32., 32., 36., 36., 36., 36.]],
                dtype=np.float32
            )
        )
        print(repr(ds.chl.data[1]))
        np.testing.assert_array_equal(
            ds.chl.data[1],
            np.array(
                [[48., 48., 48., 48., 52., 52., 52., 52.],
                 [48., 48., 48., 48., 52., 52., 52., 52.],
                 [64., 64., 64., 64., 68., 68., 68., 68.],
                 [64., 64., 64., 64., 68., 68., 68., 68.],
                 [80., 80., 80., 80., 84., 84., 84., 84.],
                 [80., 80., 80., 80., 84., 84., 84., 84.]],
                dtype=np.float32
            )
        )
        print(repr(ds.chl.data[2]))
        np.testing.assert_array_equal(
            ds.chl.data[2],
            np.array(
                [[96., 96., 96., 96., 100., 100., 100., 100.],
                 [96., 96., 96., 96., 100., 100., 100., 100.],
                 [112., 112., 112., 112., 116., 116., 116., 116.],
                 [112., 112., 112., 112., 116., 116., 116., 116.],
                 [128., 128., 128., 128., 132., 132., 132., 132.],
                 [128., 128., 128., 128., 132., 132., 132., 132.]],
                dtype=np.float32
            )
        )


class ZarrStoreTest(unittest.TestCase):
    """This test is used to assert that Zarr stores
    behave as expected with xarray, because GenericArrayStore
    expects the behavior tested here.
    """

    def setUp(self) -> None:
        self.dtype = np.dtype(np.int16)
        self.store: Dict[str, Any] = {
            ".zgroup": dict_to_bytes({
                "zarr_format": 2,
            }),
            ".zattrs": dict_to_bytes({
            }),
            "x": str_to_bytes(""),
            "x/.zarray": dict_to_bytes({
                "zarr_format": 2,
                "dtype": self.dtype.str,
                "shape": [8],
                "chunks": [4],
                "order": "C",
                "compressor": None,
                "filters": None,
                "fill_value": None,
            }),
            "x/.zattrs": dict_to_bytes({
                "_ARRAY_DIMENSIONS": ["x"],
            }),
        }

    def test_works_with_ndarray_chunks(self):
        # Here, x's chunks are numpy arrays rather than bytes!
        self.store.update({
            "x/0": np.linspace(1, 4, 4, dtype=self.dtype),
            "x/1": np.linspace(5, 8, 4, dtype=self.dtype),
        })

        ds = xr.open_zarr(self.store, consolidated=False)
        self.assertEqual(
            [1, 2, 3, 4, 5, 6, 7, 8],
            list(ds.x)
        )

    def test_works_with_bytes_chunks(self):
        # Here, x's chunks are numpy arrays rather than bytes!
        self.store.update({
            "x/0": ndarray_to_bytes(np.linspace(1, 4, 4, dtype=self.dtype)),
            "x/1": ndarray_to_bytes(np.linspace(5, 8, 4, dtype=self.dtype)),
        })

        ds = xr.open_zarr(self.store, consolidated=False)
        self.assertEqual(
            [1, 2, 3, 4, 5, 6, 7, 8],
            list(ds.x)
        )
