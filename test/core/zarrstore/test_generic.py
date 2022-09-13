# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
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

import unittest
from typing import Dict, Any

import numpy as np
import pytest
import s3fs
import xarray as xr

from test.s3test import MOTO_SERVER_ENDPOINT_URL
from test.s3test import S3Test
from xcube.core.new import new_cube
from xcube.core.zarrstore.diagnostic import DiagnosticZarrStore
from xcube.core.zarrstore.generic import GenericArray
from xcube.core.zarrstore.generic import GenericZarrStore
from xcube.core.zarrstore.generic import dict_to_bytes
from xcube.core.zarrstore.generic import get_array_slices
from xcube.core.zarrstore.generic import get_chunk_indexes
from xcube.core.zarrstore.generic import get_chunk_padding
from xcube.core.zarrstore.generic import get_chunk_shape
from xcube.core.zarrstore.generic import ndarray_to_bytes


# noinspection PyMethodMayBeStatic
class GenericArrayTest(unittest.TestCase):
    data = np.linspace(1, 4, 4)

    def get_data(self):
        return self.data

    def test_defaults(self):
        self.assertEqual({},
                         GenericArray())

    def test_finalize_converts_fill_value(self):
        data = np.linspace(1, 4, 4, dtype=np.uint16)

        # noinspection PyTypeChecker
        self.assertEqual({
            "name": "x",
            "dtype": "<u2",
            "dims": ("x",),
            "shape": (4,),
            "chunks": (4,),
            "data": data,
            "order": "C",
            "compressor": None,
            "filters": None,
            "fill_value": 0,
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
                        data=data,
                        fill_value=np.array(0)).finalize())

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

    def test_finalize_validates_name(self):
        with pytest.raises(ValueError, match="missing array name"):
            GenericArray(dims=["x"],
                         data=self.data).finalize()

    def test_finalize_validates_data_get_data(self):
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
                         data=self.data,
                         get_data=self.get_data).finalize()

        with pytest.raises(TypeError,
                           match="array 'x': get_data must be a callable"):
            # noinspection PyTypeChecker
            GenericArray(name="x",
                         dims=["x"],
                         get_data=self.data).finalize()

    def test_finalize_validates_dims(self):
        with pytest.raises(ValueError,
                           match="array 'x': missing dims"):
            GenericArray(name="x",
                         data=self.data).finalize()

    def test_finalize_validates_dtype(self):
        with pytest.raises(ValueError,
                           match="array 'x': missing dtype"):
            GenericArray(name="x",
                         dims=["x"],
                         shape=[4],
                         get_data=self.get_data).finalize()

    def test_finalize_validates_shape(self):
        with pytest.raises(ValueError,
                           match="array 'x': missing shape"):
            GenericArray(name="x",
                         dims=["x"],
                         dtype=self.data.dtype,
                         get_data=self.get_data).finalize()

        with pytest.raises(ValueError,
                           match="array 'x':"
                                 " dims and shape must have same length"):
            GenericArray(name="x",
                         dims=["x"],
                         dtype=self.data.dtype,
                         shape=[200, 300],
                         get_data=self.get_data).finalize()

    def test_finalize_validates_chunks(self):
        with pytest.raises(ValueError,
                           match="array 'x':"
                                 " dims and chunks must have same length"):
            GenericArray(name="x",
                         dims=["x"],
                         dtype=self.data.dtype,
                         shape=[300],
                         chunks=[20, 30],
                         get_data=self.get_data).finalize()

    def test_finalize_validates_filters(self):
        with pytest.raises(TypeError,
                           match="array 'x':"
                                 " filter items must be an instance"
                                 " of numcodecs.abc.Codec"):
            # noinspection PyTypeChecker
            GenericArray(name="x",
                         dims=["x"],
                         dtype=self.data.dtype,
                         shape=[300],
                         chunks=[30],
                         filters=["identity"],
                         get_data=self.get_data).finalize()

    def test_finalize_validates_compressor(self):
        with pytest.raises(TypeError,
                           match="array 'x':"
                                 " compressor must be an instance"
                                 " of numcodecs.abc.Codec"):
            # noinspection PyTypeChecker
            GenericArray(name="x",
                         dims=["x"],
                         dtype=self.data.dtype,
                         shape=[300],
                         chunks=[30],
                         compressor="blosc",
                         get_data=self.get_data).finalize()

    def test_finalize_validates_fill_value(self):
        with pytest.raises(TypeError,
                           match="array 'x':"
                                 " fill_value type must be one of"
                                 " \\('NoneType', 'bool', 'int',"
                                 " 'float', 'str'\\), was bytes"):
            # noinspection PyTypeChecker
            GenericArray(name="x",
                         dims=["x"],
                         dtype=self.data.dtype,
                         shape=[300],
                         chunks=[30],
                         fill_value=b'0123',
                         get_data=self.get_data).finalize()

    def test_finalize_validates_order(self):
        with pytest.raises(ValueError,
                           match="array 'x':"
                                 " order must be one of \\('C', 'F'\\),"
                                 " was 'D'"):
            # noinspection PyTypeChecker
            GenericArray(name="x",
                         dims=["x"],
                         dtype=self.data.dtype,
                         shape=[300],
                         chunks=[30],
                         order="D",
                         get_data=self.get_data).finalize()

    def test_finalize_validates_chunk_encoding(self):
        with pytest.raises(ValueError,
                           match="array 'x':"
                                 " chunk_encoding must be one of"
                                 " \\('bytes', 'ndarray'\\), was 'strings'"):
            # noinspection PyTypeChecker
            GenericArray(name="x",
                         dims=["x"],
                         dtype=self.data.dtype,
                         shape=[300],
                         get_data=self.get_data,
                         chunk_encoding="strings").finalize()

    def test_finalize_validates_attrs(self):
        with pytest.raises(TypeError,
                           match="array 'x': attrs must be dict, was str"):
            # noinspection PyTypeChecker
            GenericArray(name="x",
                         dims=["x"],
                         dtype=self.data.dtype,
                         shape=[300],
                         get_data=self.get_data,
                         attrs="title=x Axis").finalize()


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

        store.add_array(name="spatial_ref", dims=(),
                        data=np.array(0))

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

    def test_add_array_validates_name(self):
        store = self.new_zarr_store((3, 6, 8), (1, 2, 4), self.get_data)

        tsm = np.zeros((3, 6, 8))
        with pytest.raises(ValueError,
                           match="array 'chl' is already defined"):
            store.add_array(name="chl",
                            dims=["time", "y", "x"],
                            data=tsm)

    def test_add_array_validates_dim_sizes(self):
        store = self.new_zarr_store((3, 6, 8), (1, 2, 4), self.get_data)

        tsm = np.zeros((3, 10, 8))
        with pytest.raises(ValueError,
                           match="array 'tsm' defines"
                                 " dimension 'y' with size 10,"
                                 " but existing size is 6"):
            store.add_array(name="tsm",
                            dims=["time", "y", "x"],
                            data=tsm)

    def test_store_override_flags(self):
        store = self.new_zarr_store((3, 6, 8), (1, 2, 4), self.get_data)
        self.assertEqual(True, store.is_listable())
        self.assertEqual(True, store.is_readable())
        self.assertEqual(True, store.is_erasable())
        self.assertEqual(False, store.is_writeable())

    def test_store_override_keys(self):
        store = self.new_zarr_store((3, 6, 8), (1, 2, 4), self.get_data)
        self.assertEqual({
            '.zmetadata',
            '.zgroup',
            '.zattrs',
            'x/.zarray', 'x/.zattrs',
            'x/0',
            'y/.zarray', 'y/.zattrs',
            'y/0',
            'time/.zarray', 'time/.zattrs',
            'time/0',
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
            'spatial_ref/0',
            'spatial_ref/.zarray',
            'spatial_ref/.zattrs',
        }, set(store.keys()))

    def test_store_override_listdir(self):
        store = self.new_zarr_store((3, 6, 8), (1, 2, 4), self.get_data)
        self.assertEqual({
            '.zmetadata',
            '.zgroup',
            '.zattrs',
            'x',
            'y',
            'time',
            'chl',
            'spatial_ref',
        }, set(store.listdir('')))

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

    def test_store_override_rmdir(self):
        store = self.new_zarr_store((3, 6, 8), (1, 2, 4), self.get_data)
        store.rmdir("chl")
        self.assertEqual(
            [
                '.zattrs',
                '.zgroup',
                '.zmetadata',
                'spatial_ref',
                'time',
                'x',
                'y'
            ],
            store.listdir("")
        )

        # Also remove dimension sizes from object
        store.rmdir("x")
        store.rmdir("y")
        store.rmdir("time")

        with pytest.raises(ValueError,
                           match="chl: can only remove existing arrays"):
            store.rmdir("chl")

    def test_store_override_rename(self):
        store = self.new_zarr_store((3, 6, 8), (1, 2, 4), self.get_data)
        store.rename("chl", "chl_old")
        self.assertEqual(
            [
                '.zattrs',
                '.zgroup',
                '.zmetadata',
                'chl_old',
                'spatial_ref',
                'time',
                'x',
                'y'
            ],
            store.listdir("")
        )

        with pytest.raises(ValueError,
                           match="can only rename arrays,"
                                 " but 'tsm' is not an array"):
            store.rename("tsm", "tsm_new")

        with pytest.raises(ValueError,
                           match="cannot rename array"
                                 " 'chl_old' into 'x'"
                                 " because it already exists"):
            store.rename("chl_old", "x")

        with pytest.raises(ValueError,
                           match="cannot rename array 'chl_old'"
                                 " into 'chl/0'"):
            store.rename("chl_old", "chl/0")

    def test_store_override_close(self):
        store = self.new_zarr_store((3, 6, 8), (1, 2, 4), self.get_data)

        _array_info = None

        def handle_close(array_info):
            nonlocal _array_info
            _array_info = array_info

        tsm = np.zeros((3, 6, 8))
        store.add_array(name="tsm",
                        dims=["time", "y", "x"],
                        data=tsm,
                        on_close=handle_close)

        store.close()
        self.assertIsInstance(_array_info, dict)
        self.assertEqual("tsm", _array_info.get("name"))

    def test_store_override_iter(self):
        store = self.new_zarr_store((3, 6, 8), (1, 2, 4), self.get_data)
        self.assertEqual(set(iter(store.keys())),
                         set(iter(store)))

    def test_store_override_len(self):
        store = self.new_zarr_store((3, 6, 8), (1, 2, 4), self.get_data)
        self.assertEqual(len(list(store.keys())),
                         len(store))

    def test_store_override_contains(self):
        store = self.new_zarr_store((3, 6, 8), (1, 2, 4), self.get_data)
        self.assertTrue(".zmetadata" in store)
        self.assertTrue(".zattrs" in store)
        self.assertTrue(".zgroup" in store)
        self.assertFalse("x" in store)
        self.assertTrue("x/.zarray" in store)
        self.assertTrue("x/.zattrs" in store)
        self.assertTrue("x/0" in store)
        self.assertFalse("a" in store)
        self.assertFalse("a/0" in store)
        self.assertFalse("x/a" in store)

    def test_store_override_getitem(self):
        store = self.new_zarr_store((3, 6, 8), (1, 2, 4), self.get_data)
        self.assertIsInstance(store[".zattrs"], bytes)
        self.assertIsInstance(store["x/.zarray"], bytes)
        self.assertIsInstance(store["x/0"], bytes)

        with pytest.raises(KeyError, match="x"):
            # noinspection PyUnusedLocal
            a = store["x"]

    def test_store_override_setitem(self):
        store = self.new_zarr_store((3, 6, 8), (1, 2, 4), self.get_data)
        with pytest.raises(
                TypeError,
                match="xcube.core.zarrstore.generic.GenericZarrStore"
                      " is read-only"
        ):
            store["tsm/0.0.0"] = np.zeros((1, 2, 4)).tobytes()

    def test_store_override_delitem(self):
        store = self.new_zarr_store((3, 6, 8), (1, 2, 4), self.get_data)
        del store["x"]
        self.assertFalse("x" in store)

    def test_zarr_store_shape_not_multiple_of_chunks(self):
        shape = 3, 6, 8
        chunks = 1, 2, 5
        store = self.new_zarr_store(shape, chunks,
                                    get_data=self.get_data)

        ds = xr.open_zarr(store)

        self.assertEqual({'x', 'y', 'time'}, set(ds.coords))
        self.assertEqual({'spatial_ref', 'chl'}, set(ds.data_vars))

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
        self.assertEqual({'spatial_ref', 'chl'}, set(ds.data_vars))

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

    def test_from_dataset(self):
        store1 = self.new_zarr_store((3, 6, 8), (1, 2, 4), self.get_data)
        dataset1: xr.Dataset = xr.open_zarr(store1)

        store2 = GenericZarrStore.from_dataset(dataset1)
        self.assertIsInstance(store2, GenericZarrStore)
        self.assertIsNot(store2, store1)

        dataset2: xr.Dataset = xr.open_zarr(store2)

        xr.testing.assert_equal(dataset2, dataset1)

        dataset1.load()
        dataset2.load()
        xr.testing.assert_equal(dataset2, dataset1)


class GenericZarrStoreHelpersTest(unittest.TestCase):
    def test_get_chunk_indexes(self):
        self.assertEqual([(0,)],
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
            "x/.zarray": dict_to_bytes({
                "zarr_format": 2,
                "dtype": self.dtype.str,
                "shape": [8],
                "chunks": [4],
                "order": "C",
                "compressor": None,
                "filters": None,
                "fill_value": 7,
            }),
            "x/.zattrs": dict_to_bytes({
                "_ARRAY_DIMENSIONS": ["x"],
            }),
            "x/0": ndarray_to_bytes(np.linspace(1, 4, 4, dtype=self.dtype)),
            "x/1": ndarray_to_bytes(np.linspace(5, 8, 4, dtype=self.dtype)),
        }

    def test_works_with_bytes_chunks(self):
        ds = xr.open_zarr(self.store, consolidated=False, decode_cf=False)
        self.assertEqual(
            [1, 2, 3, 4, 5, 6, 7, 8],
            list(ds.x.values)
        )

        ds = xr.open_zarr(self.store, consolidated=False, decode_cf=True)
        np.testing.assert_array_equal(
            np.array([1., 2., 3., 4., 5., 6., float('nan'), 8.]),
            ds.x.values
        )

    def test_works_with_ndarray_chunks(self):
        # Here, x's chunks are numpy arrays rather than bytes!
        self.store.update({
            "x/0": np.linspace(1, 4, 4, dtype=self.dtype),
            "x/1": np.linspace(5, 8, 4, dtype=self.dtype),
        })

        ds = xr.open_zarr(self.store, consolidated=False, decode_cf=False)
        self.assertEqual(
            [1, 2, 3, 4, 5, 6, 7, 8],
            list(ds.x.values)
        )


class CommonS3ZarrStoreTest(S3Test):
    """This test is used to assert that the s3fs Zarr store
    behaves as expected with xarray.
    """

    def test_it(self):
        cube = new_cube(variables=dict(conc_chl=0.5)).chunk(
            dict(time=1, lat=90, lon=90)
        )

        s3 = s3fs.S3FileSystem(
            anon=False,
            client_kwargs=dict(
                endpoint_url=MOTO_SERVER_ENDPOINT_URL,
            )
        )

        s3.mkdir("xcube-test")
        s3.mkdir("xcube-test/cube.zarr")
        zarr_store = s3.get_mapper("xcube-test/cube.zarr")
        cube.to_zarr(zarr_store)

        zarr_store = s3.get_mapper("xcube-test/cube.zarr")
        zarr_store = DiagnosticZarrStore(zarr_store)

        dataset = xr.open_zarr(zarr_store)
        self.assertIsInstance(dataset, xr.Dataset)

        # print(zarr_store.records)

        self.assertIn("__getitem__('.zmetadata')", zarr_store.records)
        self.assertIn("__getitem__('lon/0')", zarr_store.records)
        self.assertIn("__getitem__('lat/0')", zarr_store.records)
        self.assertIn("__getitem__('time/0')", zarr_store.records)

        # Assert that Zarr used __getitem__ only
        for r in zarr_store.records:
            if not r.startswith("__getitem__"):
                self.fail(f"Unexpected store call: {r}")

        zarr_store.records = []
        # noinspection PyUnusedLocal
        values = dataset.conc_chl.isel(time=0).values

        # Assert that Zarr used __getitem__ only
        for r in zarr_store.records:
            if not r.startswith("__getitem__"):
                self.fail(f"Unexpected store call: {r}")
