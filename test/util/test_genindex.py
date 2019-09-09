import itertools
import json
import unittest
from abc import ABCMeta
from collections import MutableMapping
from typing import Iterator, Dict, Tuple, Iterable, KeysView

import numpy as np
import xarray as xr


class GenIndexTest(unittest.TestCase):

    def test_strict_cf_convention(self):
        index_var = gen_index_var(shape=(4, 8, 16), chunks=(2, 4, 8), dims=('time', 'lat', 'lon'))
        self.assertIsNotNone(index_var)
        self.assertEqual((4, 8, 16), index_var.shape)
        self.assertEqual(((2, 2), (4, 4), (8, 8)), index_var.chunks)
        self.assertEqual(('time', 'lat', 'lon'), index_var.dims)

        visited_indexes = set()

        def index_var_ufunc(index_var):
            nonlocal visited_indexes
            visited_indexes.add(tuple(map(int, index_var.ravel()[0:6])))
            return index_var

        result = xr.apply_ufunc(index_var_ufunc, index_var, dask='parallelized', output_dtypes=[index_var.dtype])
        self.assertIsNotNone(result)
        self.assertEqual((4, 8, 16), result.shape)
        self.assertEqual(((2, 2), (4, 4), (8, 8)), result.chunks)
        self.assertEqual(('time', 'lat', 'lon'), result.dims)

        values = result.values
        self.assertEqual((4, 8, 16), values.shape)
        np.testing.assert_array_equal(index_var.values, values)

        self.assertEqual(8, len(visited_indexes))
        self.assertEqual({
            (0, 2, 0, 4, 0, 8),
            (2, 4, 0, 4, 0, 8),
            (0, 2, 4, 8, 0, 8),
            (2, 4, 4, 8, 0, 8),
            (0, 2, 0, 4, 8, 16),
            (2, 4, 0, 4, 8, 16),
            (0, 2, 4, 8, 8, 16),
            (2, 4, 4, 8, 8, 16),
        }, visited_indexes)


def gen_index_var(shape, chunks, dims):
    ds = xr.open_zarr(IndexStore(shape, dims, chunks))
    return ds.__index_var__


def _dict_to_bytes(d: Dict):
    return _str_to_bytes(json.dumps(d, indent=2))


def _str_to_bytes(s: str):
    return bytes(s, encoding='utf-8')


class IndexStore(MutableMapping, metaclass=ABCMeta):
    """
    A index-generating Zarr Store.
    """

    def __init__(self, shape, dims, chunks, trace_store_calls=False):

        self._ndim = len(shape)
        self._shape = shape
        self._dims = dims
        self._chunks = chunks
        self._trace_store_calls = trace_store_calls

        # setup Virtual File System (vfs)
        self._vfs = {
            '.zgroup': _dict_to_bytes(dict(zarr_format=2)),
            '.zattrs': _dict_to_bytes(dict())
        }

        name = '__index_var__'
        array_metadata = dict(zarr_format=2,
                              shape=shape,
                              chunks=chunks,
                              compressor=None,
                              dtype='<u8',
                              fill_value=None,
                              filters=None,
                              order='C')
        self._vfs[name] = _str_to_bytes('')
        self._vfs[name + '/.zarray'] = _dict_to_bytes(array_metadata)
        self._vfs[name + '/.zattrs'] = _dict_to_bytes(dict(_ARRAY_DIMENSIONS=dims))

        nums = np.array(shape) // np.array(chunks)
        indexes = itertools.product(*tuple(map(range, map(int, nums))))
        for index in indexes:
            filename = '.'.join(map(str, index))
            # noinspection PyTypeChecker
            self._vfs[name + '/' + filename] = name, index

    def _fetch_chunk(self, name: str, chunk_index: Tuple[int, ...]) -> bytes:
        data = np.zeros(self._chunks, dtype=np.uint64)
        data_view = data.ravel()
        if data_view.base is not data:
            raise ValueError('view expected')
        if data_view.size < self._ndim * 2:
            raise ValueError('size too small')
        for i in range(self._ndim):
            j1 = self._chunks[i] * chunk_index[i]
            j2 = j1 + self._chunks[i]
            data_view[2 * i] = j1
            data_view[2 * i + 1] = j2
        return data.tobytes()

    @property
    def _class_name(self):
        return self.__module__ + '.' + self.__class__.__name__

    ###############################################################################
    # Zarr Store (MutableMapping) implementation
    ###############################################################################

    def keys(self) -> KeysView[str]:
        if self._trace_store_calls:
            print(f'{self._class_name}.keys()')
        return self._vfs.keys()

    def listdir(self, key: str) -> Iterable[str]:
        if self._trace_store_calls:
            print(f'{self._class_name}.listdir(key={key!r})')
        if key == '':
            return (k for k in self._vfs.keys() if '/' not in k)
        else:
            prefix = key + '/'
            start = len(prefix)
            return (k for k in self._vfs.keys() if k.startswith(prefix) and k.find('/', start) == -1)

    def getsize(self, key: str) -> int:
        if self._trace_store_calls:
            print(f'{self._class_name}.getsize(key={key!r})')
        return len(self._vfs[key])

    def __iter__(self) -> Iterator[str]:
        if self._trace_store_calls:
            print(f'{self._class_name}.__iter__()')
        return iter(self._vfs.keys())

    def __len__(self) -> int:
        if self._trace_store_calls:
            print(f'{self._class_name}.__len__()')
        return len(self._vfs.keys())

    def __contains__(self, key) -> bool:
        if self._trace_store_calls:
            print(f'{self._class_name}.__contains__(key={key!r})')
        return key in self._vfs

    def __getitem__(self, key: str) -> bytes:
        if self._trace_store_calls:
            print(f'{self._class_name}.__getitem__(key={key!r})')
        value = self._vfs[key]
        if isinstance(value, tuple):
            return self._fetch_chunk(*value)
        return value

    def __setitem__(self, key: str, value: bytes) -> None:
        if self._trace_store_calls:
            print(f'{self._class_name}.__setitem__(key={key!r}, value={value!r})')
        raise TypeError(f'{self._class_name} is read-only')

    def __delitem__(self, key: str) -> None:
        if self._trace_store_calls:
            print(f'{self._class_name}.__delitem__(key={key!r})')
        raise TypeError(f'{self._class_name} is read-only')
