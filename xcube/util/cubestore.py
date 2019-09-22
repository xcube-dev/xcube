import itertools
import json
from collections import MutableMapping
from typing import Iterator, Dict, Tuple, Iterable, KeysView, Callable, Any, Union

import numpy as np

GetChunk = Callable[["CubeStore", str, Tuple[int, ...]], bytes]


class CubeStore(MutableMapping):
    """
    A Zarr Store that generates compatible xcube datasets.
    """

    def __init__(self,
                 dims: Tuple[str, ...],
                 shape: Tuple[int, ...],
                 chunks: Tuple[int, ...],
                 attrs: Dict[str, Any] = None,
                 get_chunk: GetChunk = None,
                 trace_store_calls: bool = False):

        self._ndim = len(dims)
        self._dims = dims
        self._shape = shape
        self._chunks = chunks
        self._get_chunk = get_chunk
        self._trace_store_calls = trace_store_calls

        # setup Virtual File System (vfs)
        self._vfs = {
            '.zgroup': _dict_to_bytes(dict(zarr_format=2)),
            '.zattrs': _dict_to_bytes(attrs or dict())
        }

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def dims(self) -> Tuple[str, ...]:
        return self._dims

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def chunks(self) -> Tuple[int, ...]:
        return self._chunks

    def add_array(self, name: str, array: np.ndarray, attrs: Dict):
        shape = list(map(int, array.shape))
        dtype = str(array.dtype.str)
        array_metadata = {
            "zarr_format": 2,
            "chunks": shape,
            "shape": shape,
            "dtype": dtype,
            "fill_value": None,
            "compressor": None,
            "filters": None,
            "order": "C",
        }
        self._vfs[name] = _str_to_bytes('')
        self._vfs[name + '/.zarray'] = _dict_to_bytes(array_metadata)
        self._vfs[name + '/.zattrs'] = _dict_to_bytes(attrs)
        self._vfs[name + '/' + ('.'.join(['0'] * array.ndim))] = bytes(array)

    def add_lazy_array(self,
                       name: str,
                       dtype: str,
                       fill_value: Union[int, float] = None,
                       compressor: Dict[str, Any] = None,
                       filters=None,
                       order: str = 'C',
                       attrs: Dict[str, Any] = None,
                       get_chunk: GetChunk = None):

        get_chunk = get_chunk or self._get_chunk
        if get_chunk is None:
            raise ValueError('get_chunk must be given aas there is no default')

        array_metadata = dict(zarr_format=2,
                              shape=self._shape,
                              chunks=self._chunks,
                              compressor=compressor,
                              dtype=dtype,
                              fill_value=fill_value,
                              filters=filters,
                              order=order)

        self._vfs[name] = _str_to_bytes('')
        self._vfs[name + '/.zarray'] = _dict_to_bytes(array_metadata)
        self._vfs[name + '/.zattrs'] = _dict_to_bytes(dict(_ARRAY_DIMENSIONS=self._dims, **(attrs or dict())))

        nums = np.array(self._shape) // np.array(self._chunks)
        indexes = itertools.product(*tuple(map(range, map(int, nums))))
        for index in indexes:
            filename = '.'.join(map(str, index))
            # noinspection PyTypeChecker
            self._vfs[name + '/' + filename] = name, index, get_chunk

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
            name, index, get_chunk = value
            return get_chunk(self, name, index)
        return value

    def __setitem__(self, key: str, value: bytes) -> None:
        if self._trace_store_calls:
            print(f'{self._class_name}.__setitem__(key={key!r}, value={value!r})')
        raise TypeError(f'{self._class_name} is read-only')

    def __delitem__(self, key: str) -> None:
        if self._trace_store_calls:
            print(f'{self._class_name}.__delitem__(key={key!r})')
        raise TypeError(f'{self._class_name} is read-only')


def _dict_to_bytes(d: Dict):
    return _str_to_bytes(json.dumps(d, indent=2))


def _str_to_bytes(s: str):
    return bytes(s, encoding='utf-8')
