import collections
import functools
import itertools
from typing import Callable, Tuple, Any, Sequence, Iterable

import dask.array as da
import dask.base as db
import numpy as np

IntTuple = Tuple[int, ...]
SliceTuple = Tuple[slice, ...]
IntIterable = Iterable[int]
IntTupleIterable = Iterable[IntTuple]
SliceTupleIterable = Iterable[SliceTuple]

ChunkContext = collections.namedtuple('ChunkContext',
                                      ['chunk_shape',
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
                            *args,
                            name: str = None,
                            **kwargs) -> da.Array:
    chunk_sizes = tuple(get_chunk_sizes(shape, chunks))
    chunk_indexes, chunk_shapes, chunk_slices = get_chunk_iterators(chunk_sizes)

    name = name or 'from_func'
    array_name = name + '-' + db.tokenize(shape, chunks, dtype)

    dsk = {}
    for chunk_index, chunk_slices, chunk_shape in zip(chunk_indexes, chunk_slices, chunk_shapes):
        context = ChunkContext(chunk_index=chunk_index,
                               chunk_slices=chunk_slices,
                               chunk_shape=chunk_shape,
                               array_shape=shape,
                               array_chunks=chunk_sizes,
                               dtype=dtype,
                               name=name)
        key = (array_name,) + chunk_index
        value = (functools.partial(func, context, *args, **kwargs),)
        dsk[key] = value

    return da.Array(dsk, array_name, chunk_sizes,
                    dtype=dtype,
                    shape=shape)


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
