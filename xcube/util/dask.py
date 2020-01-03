import collections
import functools
import itertools
from typing import Callable, Tuple, Any

import dask.array as da
import dask.base as db
import numpy as np

ChunkContext = collections.namedtuple('ChunkContext',
                                      ['chunk_shape',
                                       'chunk_index',
                                       'chunk_slices',
                                       'array_shape',
                                       'array_chunks',
                                       'dtype'])


def compute_array_from_func(func: Callable[[ChunkContext], np.ndarray],
                            shape: Tuple[int, ...],
                            chunks: Tuple[int, ...],
                            dtype: Any,
                            *args,
                            name: str = None,
                            **kwargs):
    chunk_size_tuples = tuple(get_chunks(shape, chunks))
    chunk_slices_tuples = get_chunk_slices_tuples(chunk_size_tuples)
    chunk_indexes = itertools.product(*(range(len(chunk_size_tuple))
                                        for chunk_size_tuple in chunk_size_tuples))
    chunk_slices = itertools.product(*chunk_slices_tuples)
    chunk_shapes = itertools.product(*chunk_size_tuples)

    name = (name or 'from_func') + '-' + db.tokenize(shape, chunks, dtype)
    dsk = {}
    for chunk_index, chunk_slices, chunk_shape in zip(chunk_indexes, chunk_slices, chunk_shapes):
        context = ChunkContext(chunk_index=chunk_index,
                               chunk_slices=chunk_slices,
                               chunk_shape=chunk_shape,
                               array_shape=shape,
                               array_chunks=chunk_size_tuples,
                               dtype=dtype)
        key = (name,) + chunk_index
        value = (functools.partial(func, context, *args, **kwargs),)
        dsk[key] = value

    return da.Array(dsk, name, chunk_size_tuples,
                    dtype=dtype,
                    shape=shape)


def get_chunks(shape, chunks):
    for s, c in zip(shape, chunks):
        n = s // c
        if n * c < s:
            yield (c,) * n + (s % c,)
        else:
            yield (c,) * n


def get_chunk_slices_tuples(chunk_size_tuples):
    for chunk_size_tuple in chunk_size_tuples:
        yield tuple(get_chunk_slice_tuple(chunk_size_tuple))


def get_chunk_slice_tuple(chunk_size_tuple):
    stop = 0
    for i in range(len(chunk_size_tuple)):
        start = stop
        stop = start + chunk_size_tuple[i]
        yield slice(start, stop)
