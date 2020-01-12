import collections
import itertools
import uuid
from typing import Callable, Tuple, Any, Sequence, Iterable, Union, List, Mapping

import dask.array as da
import dask.array.core as dac
import numpy as np

IntTuple = Tuple[int, ...]
SliceTuple = Tuple[slice, ...]
IntIterable = Iterable[int]
IntTupleIterable = Iterable[IntTuple]
SliceTupleIterable = Iterable[SliceTuple]

ChunkContext = collections.namedtuple('ChunkContext',
                                      ['chunk_id',
                                       'chunk_shape',
                                       'chunk_index',
                                       'chunk_slices',
                                       'array_shape',
                                       'array_chunks',
                                       'dtype',
                                       'name'])


def compute_array_from_func(func: Callable[[ChunkContext], np.ndarray],
                            shape: IntTuple,
                            chunks: IntTuple,
                            dtype: Any,
                            name: str = None,
                            args: Sequence[Any] = None,
                            kwargs: Mapping[str, Any] = None) -> da.Array:
    chunk_sizes = tuple(get_chunk_sizes(shape, chunks))
    chunk_counts = tuple(get_chunk_counts(shape, chunks))
    chunk_indexes, chunk_shapes, chunk_slices = get_chunk_iterators(chunk_sizes)

    arrays = _NestedList(shape=chunk_counts)
    chunk_id = 0
    for chunk_index, chunk_slices, chunk_shape in zip(chunk_indexes, chunk_slices, chunk_shapes):
        context = ChunkContext(chunk_id=chunk_id,
                               chunk_index=chunk_index,
                               chunk_slices=chunk_slices,
                               chunk_shape=chunk_shape,
                               array_shape=shape,
                               array_chunks=chunk_sizes,
                               dtype=dtype,
                               name=name)
        chunk_id += 1

        # We use our own name here, because dac.from_func() tokenizes args which for some reason takes forever
        array = dac.from_func(func,
                              shape=chunk_shape,
                              dtype=dtype,
                              name= f'rectify_{name}-{uuid.uuid4()}',
                              args=(context, *(args or ())),
                              kwargs=(kwargs or {}))

        arrays[chunk_index] = array

    return da.block(arrays.data)


def get_chunk_iterators(chunk_sizes: IntTupleIterable) -> \
        Tuple[IntTupleIterable, IntTupleIterable, SliceTupleIterable]:
    chunk_sizes = tuple(chunk_sizes)
    chunk_slices_tuples = get_chunk_slice_tuples(chunk_sizes)
    chunk_ranges = get_chunk_ranges(chunk_sizes)
    chunk_indexes = itertools.product(*chunk_ranges)
    chunk_slices = itertools.product(*chunk_slices_tuples)
    chunk_shapes = itertools.product(*chunk_sizes)
    return chunk_indexes, chunk_shapes, chunk_slices


def get_chunk_sizes(shape: IntTuple, chunks: IntTuple) -> IntTupleIterable:
    for s, c in zip(shape, chunks):
        n = s // c
        if n * c < s:
            yield (c,) * n + (s % c,)
        else:
            yield (c,) * n


def get_chunk_counts(shape: IntTuple, chunks: IntTuple) -> Iterable[int]:
    for s, c in zip(shape, chunks):
        yield (s + c - 1) // c


def get_chunk_ranges(chunk_size_tuples: IntTupleIterable) -> Iterable[range]:
    return (range(len(chunk_size_tuple)) for chunk_size_tuple in chunk_size_tuples)


def get_chunk_slice_tuples(chunk_size_tuples: IntTupleIterable) -> SliceTupleIterable:
    return (tuple(get_chunk_slices(chunk_size_tuple)) for chunk_size_tuple in chunk_size_tuples)


def get_chunk_slices(chunk_sizes: Sequence[int]) -> Iterable[slice]:
    stop = 0
    for i in range(len(chunk_sizes)):
        start = stop
        stop = start + chunk_sizes[i]
        yield slice(start, stop)


class _NestedList:
    """
    Utility class whose instances are used as input to dask.block().
    """

    def __init__(self, shape: Sequence[int], fill_value: Any = None):
        self._shape = tuple(shape)
        self._data = self._new_data(shape, len(shape), fill_value, 0)

    @classmethod
    def _new_data(cls, shape: Sequence[int], ndim: int, fill_value: Any, dim: int) -> Union[List[List], List[Any]]:
        return [cls._new_data(shape, ndim, fill_value, dim + 1) if dim < ndim - 1 else fill_value
                for _ in range(shape[dim])]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def data(self) -> Union[List[List], List[Any]]:
        return self._data

    def __len__(self) -> int:
        return len(self._data)

    def __setitem__(self, index: Union[int, slice, tuple], value: Any):
        data = self._data
        if isinstance(index, tuple):
            n = len(index)
            for i in range(n - 1):
                data = data[index[i]]
            data[index[n - 1]] = value
        else:
            data[index] = value

    def __getitem__(self, index: Union[int, slice, tuple]) -> Any:
        data = self._data
        if isinstance(index, tuple):
            n = len(index)
            for i in range(n - 1):
                data = data[index[i]]
            return data[index[n - 1]]
        else:
            return data[index]
