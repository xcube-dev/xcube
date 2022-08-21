import itertools
import json
import math
import unittest
from collections.abc import MutableMapping
from typing import Iterator, Dict, Tuple, Iterable, \
    KeysView, Any, Sequence, Callable, Optional
from typing import Union

import numcodecs.abc
import numpy
import numpy as np
import xarray as xr

GetData = Callable[[Tuple[int], Tuple[int], Dict[str, Any]],
                   Union[bytes, numpy.ndarray]]


class ZarrStore(MutableMapping):

    def __init__(self,
                 /,
                 attrs: Optional[Dict[str, Any]] = None,
                 dims: Optional[Sequence[str]] = None,
                 shape: Optional[Sequence[int]] = None,
                 chunks: Optional[Sequence[int]] = None,
                 compressor: Optional[numcodecs.abc.Codec] = None,
                 order: Optional[str] = None,
                 get_data: Optional[GetData] = None):

        self._attrs = attrs
        self._dims = tuple(dims)
        self._shape = tuple(shape)
        self._chunks = tuple(chunks)
        self._compressor = compressor
        self._order = order
        self._get_data = get_data

        self._arrays = dict()

    def add_array(self,
                  name: str,
                  get_data: Optional[GetData] = None,
                  data: Optional[np.ndarray] = None,
                  dtype: Optional[str] = None,
                  dims: Optional[Union[str, Sequence[str]]] = None,
                  shape: Optional[Sequence[int]] = None,
                  chunks: Optional[Sequence[int]] = None,
                  fill_value: Optional[Union[int, float]] = None,
                  compressor: Optional[numcodecs.abc.Codec] = None,
                  filters=None,
                  order: Optional[str] = None,
                  attrs: Optional[Dict[str, Any]] = None):
        if data is not None and get_data is not None:
            raise ValueError()
        if data is None and get_data is None:
            if self._get_data is None:
                raise ValueError()
            get_data = self._get_data

        dims = dims or self._dims
        dims = (dims,) if isinstance(dims, str) else tuple(dims)
        if isinstance(data, np.ndarray):
            assert len(dims) == data.ndim
            dtype = str(data.dtype.str)
            shape = shape or tuple(map(int, data.shape))
            chunks = chunks or shape
            if chunks != shape:
                raise ValueError("arrays with data cannot yet be chunked")

        # TODO: verify
        shape = shape or self._shape
        chunks = chunks or self._chunks
        order = order or self._order or "C"
        attrs = dict(attrs) if attrs else None
        compressor = compressor or self._compressor or None

        num_chunks = tuple(map(lambda x: math.ceil(x[0] / x[1]),
                               zip(shape, chunks)))

        self._arrays[name] = {
            "name": name,
            "dtype": dtype,
            "ndim": len(dims),
            "dims": tuple(dims),
            "shape": tuple(shape),
            "chunks": tuple(chunks),
            "num_chunks": num_chunks,
            "compressor": compressor,
            "fill_value": fill_value,
            "filters": filters,
            "order": order,
            "attrs": attrs,
            "data": data,
            "get_data": get_data
        }

    ##########################################################################
    # Special Zarr helpers implementation
    ##########################################################################

    def keys(self) -> KeysView[str]:
        yield ".zmetadata"
        yield ".zgroup"
        yield ".zattrs"
        for array_name, array_info in self._arrays.items():
            yield array_name
            yield from self._get_array_keys(array_name)

    def listdir(self, key: str) -> Iterable[str]:
        if key == "":
            yield ".zmetadata"
            yield ".zgroup"
            yield ".zattrs"
            for array_name, array_info in self._arrays.items():
                yield array_name
        elif "/" not in key:
            yield from self._get_array_keys(key)

    # Note, getsize is not implemented by intention as it requires
    # actual computation of arrays.
    #
    # def getsize(self, key: str) -> int:
    #     pass

    ##########################################################################
    # Mapping implementation
    ##########################################################################

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def __contains__(self, key: str) -> bool:
        if key in (".zmetadata", ".zgroup", ".zattrs"):
            return True
        if key in self._arrays:
            return True
        try:
            array_name, value_id = self._parse_array_key(key)
        except KeyError:
            return False
        if value_id in (".zarray", ".zattrs"):
            return True
        try:
            self._get_array_chunk_index(array_name, value_id)
            return True
        except KeyError:
            return False

    def __getitem__(self, key: str) -> bytes:
        item = self._get_item(key)
        if isinstance(item, dict):
            return dict_to_bytes(item)
        if isinstance(item, str):
            return str_to_bytes(item)
        assert isinstance(item, bytes)
        return item

    ##########################################################################
    # MutableMapping implementation
    ##########################################################################

    def __setitem__(self, key: str, value: bytes) -> None:
        raise TypeError(f'{self._class_name} is read-only')

    def __delitem__(self, key: str) -> None:
        raise TypeError(f'{self._class_name} is read-only')

    ##########################################################################
    # Helpers
    ##########################################################################

    @property
    def _class_name(self):
        return self.__module__ + '.' + self.__class__.__name__

    def _get_item(self, key: str) -> Union[dict, str, bytes]:
        if key == ".zmetadata":
            return self._get_metadata_item()
        if key == ".zgroup":
            return self._get_group_item()
        if key == ".zattrs":
            return self._get_attrs_item()
        if key in self._arrays:
            return ""

        array_name, value_id = self._parse_array_key(key)
        array_info = self._arrays[array_name]

        if value_id == '.zarray':
            return self._get_array_spec_item(array_info)
        if value_id == '.zattrs':
            return self._get_array_attrs_item(array_info)

        chunk_index = self._get_array_chunk_index(array_name, value_id)
        return self._get_array_data_item(array_info, chunk_index)

    def _get_metadata_item(self):
        metadata = {
            ".zgroup": self._get_item(".zgroup"),
            ".zattrs": self._get_item(".zattrs"),
        }
        for array_name in self._arrays.keys():
            key = array_name + "/.zarray"
            metadata[key] = self._get_item(key)
            key = array_name + "/.zattrs"
            metadata[key] = self._get_item(key)
        return {
            "zarr_consolidated_format": 1,
            "metadata": metadata
        }

    # noinspection PyMethodMayBeStatic
    def _get_group_item(self):
        return {
            "zarr_format": 2
        }

    def _get_attrs_item(self):
        return self._attrs or {}

    # noinspection PyMethodMayBeStatic
    def _get_array_spec_item(self, array_info):

        compressor = array_info["compressor"]
        if compressor is not None:
            compressor = compressor.get_config()

        fill_value = array_info["fill_value"]
        if fill_value is not None:
            if isinstance(fill_value, np.ndarray):
                fill_value = fill_value.item()
            if isinstance(fill_value, float):
                if math.isnan(fill_value):
                    fill_value = "NaN"
                elif math.isinf(fill_value):
                    if fill_value < 0:
                        fill_value = "-Infinity"
                    else:
                        fill_value = "Infinity"

        return {
            "zarr_format": 2,
            "dtype": array_info["dtype"],
            "shape": list(array_info["shape"]),
            "chunks": list(array_info["chunks"]),
            "fill_value": fill_value,
            "compressor": compressor,
            "order": array_info["order"],
            "filters": array_info["filters"],
        }

    # noinspection PyMethodMayBeStatic
    def _get_array_attrs_item(self, array_info):
        dims = array_info["dims"]
        attrs = array_info["attrs"]
        return {
            "_ARRAY_DIMENSIONS": dims,
            **(attrs or {})
        }

    # noinspection PyMethodMayBeStatic
    def _get_array_data_item(self,
                             array_info: Dict[str, Any],
                             chunk_index: Tuple[int]):

        data = array_info["data"]
        if data is None:
            get_data = array_info["get_data"]
            assert callable(get_data)
            shape = array_info["shape"]
            chunks = array_info["chunks"]
            chunk_shape = get_chunk_shape(shape, chunks, chunk_index)
            data = get_data(chunk_index, chunk_shape, array_info)

        if isinstance(data, np.ndarray):
            order = array_info["order"]
            data = data.tobytes(order=order)
            compressor = array_info["compressor"]
            if compressor is not None:
                data = compressor.encode(data)

        return data

    def _parse_array_key(self, key: str) -> Tuple[str, str]:
        array_name_and_value_id = key.rsplit('/', maxsplit=1)
        if len(array_name_and_value_id) != 2:
            raise KeyError(key)
        array_name, value_id = array_name_and_value_id
        if array_name not in self._arrays:
            raise KeyError(key)
        return array_name, value_id

    def _get_array_chunk_index(self,
                               array_name: str,
                               index_id: str) -> Tuple[int]:
        try:
            chunk_index = tuple(map(int, index_id.split('.')))
        except (ValueError, TypeError):
            raise KeyError(array_name + "/" + index_id)
        array_info = self._arrays[array_name]
        shape = array_info["shape"]
        if len(chunk_index) != len(shape):
            raise KeyError(array_name + "/" + index_id)
        num_chunks = array_info["num_chunks"]
        for i, n in zip(chunk_index, num_chunks):
            if not (0 <= i < n):
                raise KeyError(array_name + "/" + index_id)
        return chunk_index

    def _get_array_keys(self, array_name):
        yield array_name + "/.zarray"
        yield array_name + "/.zattrs"
        array_info = self._arrays[array_name]
        num_chunks = array_info["num_chunks"]
        yield from self._get_array_chunk_keys(array_name, num_chunks)

    @classmethod
    def _get_array_chunk_keys(cls,
                              array_name: str,
                              num_chunks: Tuple[int]):
        for index in cls._get_array_chunk_indexes(num_chunks):
            yield array_name + '/' + '.'.join(map(str, index))

    @classmethod
    def _get_array_chunk_indexes(cls,
                                 num_chunks: Tuple[int]):
        return itertools.product(*tuple(map(range, map(int, num_chunks))))


def dict_to_bytes(d: Dict):
    return str_to_bytes(json.dumps(d, indent=2))


def str_to_bytes(s: str):
    return bytes(s, encoding='utf-8')


def get_chunk_shape(shape, chunks, chunk_index):
    chunk_shape = []
    for s, c, i in zip(shape, chunks, chunk_index):
        if (i + 1) * c > s:
            chunk_shape.append(s % c)
        else:
            chunk_shape.append(c)
    return tuple(chunk_shape)


class ZarrStoreTest(unittest.TestCase):
    def new_zarr_store(self, shape, chunks, get_data):
        store = ZarrStore(
            dims=("time", "y", "x"),
            shape=shape,
            chunks=chunks,
        )

        t_size, y_size, x_size = shape
        store.add_array('x', data=np.linspace(0., 1., x_size), dims="x")
        store.add_array('y', data=np.linspace(0., 1., y_size), dims="y")
        store.add_array('time', data=np.linspace(1, 365, t_size), dims="time")

        store.add_array("chl",
                        dtype=np.dtype(np.float32).str,
                        get_data=get_data)

        return store

    def setUp(self) -> None:
        self.chunk_shapes = set()
        self.chunk_indexes = set()

    def get_data(self, chunk_index, chunk_shape, array_info):
        st, sy, sx = array_info["shape"]
        nt, ny, nx = array_info["chunks"]
        it, iy, ix = chunk_index
        pt = it * nt
        py = iy * ny
        px = ix * nx
        value = (pt * sy + py) * sx + px
        self.chunk_shapes.add(chunk_shape)
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
        self.assertEqual({
            '.zmetadata',
            '.zgroup',
            '.zattrs',
            'x',
            'y',
            'time',
            'chl',
        }, set(store.listdir('')))

        self.assertEqual({
            'time/.zarray',
            'time/.zattrs',
            'time/0',
        }, set(store.listdir('time')))

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
