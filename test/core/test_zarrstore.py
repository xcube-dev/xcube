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

import os.path
import unittest
from typing import Dict, Any

import numpy as np
import pytest
import xarray as xr

from xcube.core.zarrstore import GenericArray
from xcube.core.zarrstore import GenericZarrStore
from xcube.core.zarrstore import XarrayZarrStore
from xcube.core.zarrstore import dict_to_bytes
from xcube.core.zarrstore import get_array_slices
from xcube.core.zarrstore import get_chunk_indexes
from xcube.core.zarrstore import get_chunk_padding
from xcube.core.zarrstore import get_chunk_shape
from xcube.core.zarrstore import ndarray_to_bytes
from xcube.core.zarrstore import str_to_bytes


# noinspection PyMethodMayBeStatic
class GenericArrayTest(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual({},
                         GenericArray())

    def test_finalize_ok_with_data(self):
        data = np.linspace(1, 4, 4)

        self.assertEqual({
            "name": "x",
            "dtype": "<f8",
            "dims": ("x",),
            "shape": (4,),
            "chunks": (4,),
            "data": data,
            "order": "C",
            "compressor": None,
            "filters": None,
            "fill_value": None,
            "get_data": None,
            "get_data_info": None,
            "get_data_params": None,
            "on_close": None,
            "chunk_encoding": "bytes",
            "attrs": None,
            # Computed
            "ndim": 1,
            "num_chunks": (1,),
        }, GenericArray(name="x",
                        dims=["x"],
                        data=data).finalize())

    def test_finalize_ok_with_get_data(self):
        shape = (12,)
        chunks = (5,)

        data = np.linspace(1, 5, 5, dtype=np.uint8)

        def get_data_1():
            return data

        self.assertEqual({
            "name": "x",
            "dtype": "|u1",
            "dims": ("x",),
            "shape": shape,
            "chunks": chunks,
            "data": None,
            "get_data": get_data_1,
            "get_data_info": {
                "has_chunk_info": False,
                "has_array_info": False
            },
            "get_data_params": {},
            "order": "C",
            "compressor": None,
            "filters": None,
            "fill_value": None,
            "on_close": None,
            "chunk_encoding": "ndarray",
            "attrs": {"units": "degrees"},
            # Computed
            "ndim": 1,
            "num_chunks": (3,),
        }, GenericArray(name="x",
                        dims=["x"],
                        dtype=data.dtype,
                        shape=shape,
                        chunks=chunks,
                        get_data=get_data_1,
                        chunk_encoding="ndarray",
                        attrs={"units": "degrees"}).finalize())

        # noinspection PyUnusedLocal
        def get_data_2(array_info=None):
            return data

        self.assertEqual({
            "name": "x",
            "dtype": "|u1",
            "dims": ("x",),
            "shape": shape,
            "chunks": chunks,
            "data": None,
            "get_data": get_data_2,
            "get_data_info": {
                "has_chunk_info": False,
                "has_array_info": True
            },
            "get_data_params": {},
            "order": "C",
            "compressor": None,
            "filters": None,
            "fill_value": None,
            "on_close": None,
            "chunk_encoding": "ndarray",
            "attrs": {"units": "degrees"},
            # Computed
            "ndim": 1,
            "num_chunks": (3,),
        }, GenericArray(name="x",
                        dims=["x"],
                        dtype=data.dtype,
                        shape=shape,
                        chunks=chunks,
                        get_data=get_data_2,
                        chunk_encoding="ndarray",
                        attrs={"units": "degrees"}).finalize())

        # noinspection PyUnusedLocal
        def get_data_3(chunk_info=None,
                       array_info=None,
                       user_data=None):
            return data

        self.assertEqual({
            "name": "x",
            "dtype": "|u1",
            "dims": ("x",),
            "shape": shape,
            "chunks": chunks,
            "data": None,
            "get_data": get_data_3,
            "get_data_info": {
                "has_chunk_info": True,
                "has_array_info": True
            },
            "get_data_params": {"user_data": 42},
            "order": "C",
            "compressor": None,
            "filters": None,
            "fill_value": None,
            "on_close": None,
            "chunk_encoding": "ndarray",
            "attrs": {"units": "degrees"},
            # Computed
            "ndim": 1,
            "num_chunks": (3,),
        }, GenericArray(name="x",
                        dims=["x"],
                        dtype=data.dtype,
                        shape=shape,
                        chunks=chunks,
                        get_data=get_data_3,
                        get_data_params=dict(user_data=42),
                        chunk_encoding="ndarray",
                        attrs={"units": "degrees"}).finalize())

    def test_finalize_raises(self):
        data = np.linspace(1, 4, 4)

        def get_data():
            return data

        with pytest.raises(ValueError, match="missing array name"):
            GenericArray(dims=["x"],
                         data=data).finalize()

        with pytest.raises(ValueError,
                           match="array 'x':"
                                 " either data or get_data must be defined"):
            GenericArray(name="x",
                         dims=["x"]).finalize()

        with pytest.raises(ValueError,
                           match="array 'x':"
                                 " data and get_data"
                                 " cannot be defined together"):
            GenericArray(name="x",
                         dims=["x"],
                         data=data,
                         get_data=get_data).finalize()

        with pytest.raises(TypeError,
                           match="array 'x': get_data must be a callable"):
            # noinspection PyTypeChecker
            GenericArray(name="x",
                         dims=["x"],
                         get_data=data).finalize()

        with pytest.raises(ValueError,
                           match="array 'x': missing dims"):
            GenericArray(name="x",
                         data=data).finalize()

        with pytest.raises(ValueError,
                           match="array 'x': missing dtype"):
            GenericArray(name="x",
                         dims=["x"],
                         shape=[4],
                         get_data=get_data).finalize()

        with pytest.raises(ValueError,
                           match="array 'x': missing shape"):
            GenericArray(name="x",
                         dims=["x"],
                         dtype=data.dtype,
                         get_data=get_data).finalize()


class GenericZarrStoreTest(unittest.TestCase):
    @staticmethod
    def new_zarr_store(shape, chunks, get_data) -> GenericZarrStore:
        store = GenericZarrStore(
            array_defaults=GenericArray(
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

    def get_data(self, chunk_info=None, array_info=None):
        chunk_index = chunk_info["index"]
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

    def test_zarr_store_shape_not_multiple_of_chunks(self):
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

    def test_zarr_store_shape_multiple_of_chunks(self):
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


class GenericZarrStoreHelpersTest(unittest.TestCase):
    def test_get_chunk_indexes(self):
        self.assertEqual([()],
                         list(get_chunk_indexes(())))
        self.assertEqual([(0,), (1,), (2,), (3,)],
                         list(get_chunk_indexes((4,))))
        self.assertEqual([(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3),
                          (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3),
                          (0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 2, 3),
                          (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3),
                          (1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 1, 3),
                          (1, 2, 0), (1, 2, 1), (1, 2, 2), (1, 2, 3)],
                         list(get_chunk_indexes((2, 3, 4))))

    def test_get_chunk_shape(self):
        shape = (3, 6, 12)
        chunks = (1, 3, 4)
        self.assertEqual(chunks,
                         get_chunk_shape(shape,
                                         chunks,
                                         (0, 0, 0)))
        self.assertEqual(chunks,
                         get_chunk_shape(shape,
                                         chunks,
                                         (2, 1, 2)))

        chunks = (1, 4, 8)
        self.assertEqual(chunks,
                         get_chunk_shape(shape,
                                         chunks,
                                         (0, 0, 0)))
        self.assertEqual((1, 2, 4),
                         get_chunk_shape(shape,
                                         chunks,
                                         (2, 1, 2)))

    def test_get_array_slices(self):
        shape = (3, 6, 12)
        chunks = (1, 3, 4)
        self.assertEqual((slice(0, 1), slice(0, 3), slice(0, 4)),
                         get_array_slices(shape,
                                          chunks,
                                          (0, 0, 0)))
        self.assertEqual((slice(2, 3), slice(3, 6), slice(8, 12)),
                         get_array_slices(shape,
                                          chunks,
                                          (2, 1, 2)))

        chunks = (1, 4, 8)
        self.assertEqual((slice(0, 1), slice(0, 4), slice(0, 8)),
                         get_array_slices(shape,
                                          chunks,
                                          (0, 0, 0)))
        self.assertEqual((slice(2, 3), slice(4, 6), slice(16, 20)),
                         get_array_slices(shape,
                                          chunks,
                                          (2, 1, 2)))

    def test_get_chunk_padding(self):
        shape = (3, 6, 12)
        chunks = (1, 3, 4)
        self.assertEqual(((0, 0), (0, 0), (0, 0)),
                         get_chunk_padding(shape,
                                           chunks,
                                           (0, 0, 0)))
        self.assertEqual(((0, 0), (0, 0), (0, 0)),
                         get_chunk_padding(shape,
                                           chunks,
                                           (2, 1, 2)))

        chunks = (1, 4, 8)
        self.assertEqual(((0, 0), (0, 0), (0, 0)),
                         get_chunk_padding(shape,
                                           chunks,
                                           (0, 0, 0)))

        chunks = (1, 4, 8)
        self.assertEqual(((0, 0), (0, 2), (0, 4)),
                         get_chunk_padding(shape,
                                           chunks,
                                           (0, 1, 1)))


class CommonZarrStoreTest(unittest.TestCase):
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


CMEMS_CREDENTIALS_FILE = "cmems-credentials.json"


# noinspection PyPackageRequirements
@unittest.skipUnless(os.path.exists(CMEMS_CREDENTIALS_FILE),
                     f"file not found: {CMEMS_CREDENTIALS_FILE}")
class PydapTest(unittest.TestCase):
    """We use this test to debug into the details of the
    pydap implementation."""

    def test_it(self):
        import json

        with open(CMEMS_CREDENTIALS_FILE) as fp:
            credentials = json.load(fp)
            username, password = credentials["username"], credentials[
                "password"]

        import pydap.client
        import pydap.cas.get_cookies

        cas_url = 'https://cmems-cas.cls.fr/cas/login'
        session = pydap.cas.get_cookies.setup_session(
            cas_url,
            username,
            password
        )
        session.cookies.set("CASTGC", session.cookies.get_dict()['CASTGC'])

        pyd_dataset = pydap.client.open_url(
            "https://nrt.cmems-du.eu/thredds/dodsC/"
            "dataset-bal-analysis-forecast-wav-hourly",
            session=session,
            user_charset='utf-8',
            # retrieve only main arrays and never retrieve coordinate axes
            output_grid=False,
        )

        data = pyd_dataset.lon[0:10]
        data = np.array(data)
        self.assertIsInstance(data, np.ndarray)


class XarrayZarrStoreTest(unittest.TestCase):
    # noinspection PyMethodMayBeStatic
    def test_it(self):
        nx = 20
        ny = 10
        nt = 5
        x = xr.DataArray(np.linspace(0, nx / 10, nx), dims="x")
        y = xr.DataArray(np.linspace(0, ny / 10, ny), dims="y")
        time = xr.DataArray(np.linspace(1, nt, nt), dims="time")
        chl = xr.DataArray(np.linspace(1, nt * ny * nx, nt * ny * nx)
                           .reshape((nt, ny, nx)),
                           dims=["time", "y", "x"]).chunk(
            (1, ny // 2, nx // 2))
        dataset = xr.Dataset(data_vars=dict(chl=chl),
                             coords=dict(time=time, y=y, x=x),
                             attrs=dict(title="Conversion test"))

        zarr_store = XarrayZarrStore(dataset)

        dataset2 = xr.open_zarr(zarr_store)

        xr.testing.assert_equal(dataset, dataset2)

        dataset.load()
        dataset2.load()

        xr.testing.assert_equal(dataset, dataset2)
