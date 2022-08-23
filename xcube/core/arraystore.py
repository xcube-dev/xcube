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

import itertools
import json
import math
from typing import Iterator, Dict, Tuple, KeysView, Any, Callable, \
    Optional, List, Sequence
from typing import Union

import numcodecs.abc
import numpy
import numpy as np
import zarr.storage

GetData = Callable[[Tuple[int], Tuple[int], "GenericArrayInfo"],
                   Union[bytes, numpy.ndarray]]

OnClose = Callable[["GenericArrayInfo"], None]


class GenericArrayInfo(dict[str, any]):
    """
    Holds information about a generic array in the ``GenericArrayStore``.

    Note that if the value of a named keyword argument is None,
    it will not be stored.

    :param array_info: Optional array info dictionary
    :param name: Optional array name
    :param get_data: Optional array data chunk getter.
        Mutually exclusive with *data*.
    :param data: Optional array data.
        Mutually exclusive with *get_data*.
    :param dtype: Optional array data type.
        Either a string using syntax of the Zarr spec or a ``numpy.dtype``.
    :param dims: Optional sequence of dimension names.
    :param shape: Optional sequence of shape sizes for each dimension.
    :param chunks: Optional sequence of chunk sizes for each dimension.
    :param fill_value: Optional fill value (int or float scalar).
    :param compressor: Optional compressor.
        If given, it must be an instance of ``numcodecs.abc.Codec``.
    :param filters: Optional filters, see Zarr spec.
    :param order: Optional array endian ordering. Defaults to "C".
    :param attrs: Optional array attributes.
        If given, must be JSON-serializable.
    :param on_close: Optional array close handler.
        Called if the store is closed.
    :param kwargs: Other keyword arguments passed directly to the
        dictionary constructor.
    """

    def __init__(self,
                 array_info: Optional[Dict[str, any]] = None,
                 name: Optional[str] = None,
                 get_data: Optional[GetData] = None,
                 data: Optional[np.ndarray] = None,
                 dtype: Optional[Union[str, np.dtype]] = None,
                 dims: Optional[Union[str, Sequence[str]]] = None,
                 shape: Optional[Sequence[int]] = None,
                 chunks: Optional[Sequence[int]] = None,
                 fill_value: Optional[Union[int, float]] = None,
                 compressor: Optional[numcodecs.abc.Codec] = None,
                 filters=None,
                 order: Optional[str] = None,
                 attrs: Optional[Dict[str, Any]] = None,
                 on_close: Optional[OnClose] = None,
                 **kwargs):
        array_info = dict(array_info) if array_info is not None else dict()
        array_info.update({
            k: v
            for k, v in dict(
                name=name,
                dtype=dtype,
                dims=dims,
                shape=shape,
                chunks=chunks,
                fill_value=fill_value,
                compressor=compressor,
                filters=filters,
                order=order,
                attrs=attrs,
                data=data,
                get_data=get_data,
                on_close=on_close
            ).items()
            if v is not None
        })
        super().__init__(array_info, **kwargs)

    def finalize(self) -> "GenericArrayInfo":
        """Normalize and validate array properties and return a valid
        array info dictionary.
        """
        name = self.get("name")
        if not name:
            raise ValueError("missing array name")

        data = self.get("data")
        get_data = self.get("get_data")
        if data is not None and get_data is not None:
            raise ValueError(f"array {name!r}:"
                             f" data and get_data cannot be defined together")
        if data is None and get_data is None:
            raise ValueError(f"array {name!r}:"
                             f" either data or get_data must be defined")

        dims = self.get("dims")
        dims = [dims] if isinstance(dims, str) else dims
        if dims is None:
            raise ValueError(f"array {name!r}: missing dims")
        ndim = len(dims)
        if isinstance(data, np.ndarray):
            # forman: maybe warn if dtype or shape is given,
            #   but does not match data.dtype and data.shape
            dtype = str(data.dtype.str)
            shape = data.shape
            chunks = data.shape
        else:
            dtype = self.get("dtype")
            shape = self.get("shape")
            chunks = self.get("chunks", shape)

        if not dtype:
            raise ValueError(f"array {name!r}: missing dtype")
        elif isinstance(dtype, np.dtype):
            dtype = dtype.str

        if shape is None:
            raise ValueError(f"array {name!r}: missing shape")
        if len(shape) != ndim:
            raise ValueError(f"array {name!r}:"
                             f" dims and shape must have same length")
        if chunks is None:
            raise ValueError(f"array {name!r}: missing chunks")
        if len(chunks) != ndim:
            raise ValueError(f"array {name!r}:"
                             f" dims and chunks must have same length")

        num_chunks = tuple(map(lambda x: math.ceil(x[0] / x[1]),
                               zip(shape, chunks)))

        compressor = self.get("compressor")
        if compressor is not None:
            if not isinstance(compressor, numcodecs.abc.Codec):
                raise TypeError(f"array {name!r}:"
                                f" compressor must be an"
                                f" instance of numcodecs.abc.Codec")

        fill_value = self.get("fill_value")
        if isinstance(fill_value, np.ndarray):
            fill_value = fill_value.item()
        if fill_value is not None \
                and not isinstance(fill_value, (int, float)):
            raise TypeError(f"array {name!r}:"
                            f" fill_value must be an"
                            f" instance of int or float")
        order = self.get("order") or "C"
        if order not in ("C", "F"):
            raise ValueError(f"array {name!r}:"
                             f" order must be 'C' or 'F', was {order!r}")

        # Note: passing the properties as dictionary
        # will prevent removing them if their value is None,
        # see ArrayInfo constructor.
        return GenericArrayInfo({
            "name": name,
            "dtype": dtype,
            "dims": tuple(dims),
            "shape": tuple(shape),
            "chunks": tuple(chunks),
            "compressor": compressor,
            "fill_value": fill_value,
            "filters": self.get("filters"),
            "order": order,
            "attrs": dict(self.get("attrs") or {}),
            "data": data,
            "get_data": get_data,
            "on_close": self.get("on_close"),
            # Computed properties
            "ndim": len(dims),
            "num_chunks": num_chunks,
        })


class GenericArrayStore(zarr.storage.Store):
    """A Zarr store that maintains generic arrays in a flat top-level
    hierarchy. The root of the store is a Zarr group
    conforming to the Zarr spec v2.

    It is designed to serve as a Zarr store for xarray datasets
    that compute their data arrays dynamically.

    The array data of this store's arrays are either retrieved from
    static (numpy) arrays or from a callable that provides the
    array's data chunks.

    :param attrs: Optional attributes of the top-level group.
        If given, it must be JSON serializable.
    :param array_defaults: Optional array defaults for
        array properties not passed to ``add_array``.
    """

    def __init__(
            self,
            attrs: Optional[Dict[str, Any]] = None,
            array_defaults: Union[None,
                                  GenericArrayInfo,
                                  Dict[str, Any]] = None
    ):
        self._attrs = dict(attrs) if attrs is not None else {}
        self._array_defaults = array_defaults
        self._dim_sizes: Dict[str, int] = {}
        self._array_infos: Dict[str, GenericArrayInfo] = {}

    def add_array(self,
                  array_info: Union[None, Dict[str, Any]] = None,
                  **array_info_kwargs) -> None:
        """
        Add a new generic array to this store.

        :param array_info: Optional array properties.
            Usually an instance of ``GenericArrayInfo``.
        :param array_info_kwargs: Keyword arguments of ``GenericArrayInfo``.
        """
        effective_array_info = GenericArrayInfo(self._array_defaults or {})
        if array_info:
            effective_array_info.update(**array_info)
        if array_info_kwargs:
            effective_array_info.update(**array_info_kwargs)
        effective_array_info = effective_array_info.finalize()

        name = effective_array_info["name"]
        if name in self._array_infos:
            raise ValueError(f"array {name!r} is already defined")

        dims = effective_array_info["dims"]
        shape = effective_array_info["shape"]
        for dim_name, dim_size in zip(dims, shape):
            old_dim_size = self._dim_sizes.get(dim_name)
            if old_dim_size is None:
                self._dim_sizes[name] = dim_size
            elif old_dim_size != dim_size:
                raise ValueError(f"array {name!r}"
                                 f" defines dimension {dim_name!r}"
                                 f" with size {dim_size},"
                                 f" but existing size is {old_dim_size}")

        self._array_infos[name] = effective_array_info

    ##########################################################################
    # Zarr Store implementation
    ##########################################################################

    def is_writeable(self):
        """Return False, because arrays in this store are generative."""
        return False

    def keys(self) -> KeysView[str]:
        """Get an iterator of all keys in this store."""
        yield ".zmetadata"
        yield ".zgroup"
        yield ".zattrs"
        for array_name, array_info in self._array_infos.items():
            yield array_name
            yield from self._get_array_keys(array_name)

    def listdir(self, path: str = "") -> List[str]:
        """List a store path.
        :param path: The path.
        :return: List of directory entries.
        """
        if path == "":
            return [
                ".zmetadata",
                ".zgroup",
                ".zattrs",
                *self._array_infos.keys()
            ]
        elif "/" not in path:
            return list(self._get_array_keys(path))
        raise ValueError(f"{path} is not a directory")

    def rmdir(self, path: str = "") -> None:
        """The general form removes store paths.
        This implementation can remove entire arrays only.
        :param path: The array's name.
        """
        if path not in self._array_infos:
            raise ValueError(f"{path}: can only remove arrays")
        array_info = self._array_infos.pop(path)
        dims = array_info["dims"]
        for i, dim_name in enumerate(dims):
            dim_used = False
            for array_name, array_info in self._array_infos.items():
                if dim_name in array_info["dims"]:
                    dim_used = True
                    break
            if not dim_used:
                del self._dim_sizes[dim_name]

    def rename(self, src_path: str, dst_path: str) -> None:
        """The general form renames store paths.
        This implementation can rename arrays only.

        :param src_path: Source array name.
        :param dst_path: Target array name.
        """
        array_info = self._array_infos.get(src_path)
        if array_info is None:
            raise ValueError(f"can only rename arrays, but {src_path!r}"
                             f" is not an array")
        if dst_path in self._array_infos:
            raise ValueError(f"cannot rename array {src_path!r} into "
                             f" into {dst_path!r} because it already exists")
        if "/" in dst_path:
            raise ValueError(f"cannot rename array {src_path!r}"
                             f" into {dst_path!r}")
        array_info["name"] = dst_path
        self._array_infos[dst_path] = array_info
        del self._array_infos[src_path]

    def close(self) -> None:
        """Calls the "on_close" handlers, if any, of arrays."""
        for array_info in self._array_infos.values():
            on_close = array_info.get("on_close")
            if on_close is not None:
                on_close(array_info)

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
        if key in self._array_infos:
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
        elif isinstance(item, str):
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
    def _class_name(self) -> str:
        return self.__module__ + '.' + self.__class__.__name__

    def _get_item(self, key: str) -> Union[dict, str, bytes]:
        if key == ".zmetadata":
            return self._get_metadata_item()
        if key == ".zgroup":
            return self._get_group_item()
        if key == ".zattrs":
            return self._get_attrs_item()
        if key in self._array_infos:
            return ""

        array_name, value_id = self._parse_array_key(key)
        array_info = self._array_infos[array_name]

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
        for array_name in self._array_infos.keys():
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
        if fill_value is not None and isinstance(fill_value, float):
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
        chunks = array_info["chunks"]
        if data is None:
            get_data = array_info["get_data"]
            assert callable(get_data)
            shape = array_info["shape"]
            chunk_shape = get_chunk_shape(shape, chunks, chunk_index)
            data = get_data(chunk_index, chunk_shape, array_info)

        if isinstance(data, np.ndarray):
            if chunks != data.shape:
                key = format_array_chunk_key(array_info["name"],
                                             chunk_index)
                raise ValueError(f"{key}:"
                                 f" data chunk must have shape {chunks},"
                                 f" but was {data.shape}")
            order = array_info["order"]
            data = data.tobytes(order=order)
            compressor = array_info["compressor"]
            if compressor is not None:
                data = compressor.encode(data)
        elif not isinstance(data, bytes):
            key = format_array_chunk_key(array_info["name"],
                                         chunk_index)
            raise TypeError(f"{key}:"
                            f" data must be of type bytes,"
                            f" but was {type(data).__name__}")

        return data

    def _parse_array_key(self, key: str) -> Tuple[str, str]:
        array_name_and_value_id = key.rsplit('/', maxsplit=1)
        if len(array_name_and_value_id) != 2:
            raise KeyError(key)
        array_name, value_id = array_name_and_value_id
        if array_name not in self._array_infos:
            raise KeyError(key)
        return array_name, value_id

    def _get_array_chunk_index(self,
                               array_name: str,
                               index_id: str) -> Tuple[int]:
        try:
            chunk_index = tuple(map(int, index_id.split('.')))
        except (ValueError, TypeError):
            raise KeyError(array_name + "/" + index_id)
        array_info = self._array_infos[array_name]
        shape = array_info["shape"]
        if len(chunk_index) != len(shape):
            raise KeyError(array_name + "/" + index_id)
        num_chunks = array_info["num_chunks"]
        for i, n in zip(chunk_index, num_chunks):
            if not (0 <= i < n):
                raise KeyError(array_name + "/" + index_id)
        return chunk_index

    def _get_array_keys(self, array_name: str) -> Iterator[str]:
        yield array_name + "/.zarray"
        yield array_name + "/.zattrs"
        array_info = self._array_infos[array_name]
        num_chunks = array_info["num_chunks"]
        yield from get_array_chunk_keys(array_name, num_chunks)


def get_chunk_shape(shape: Tuple[int],
                    chunks: Tuple[int],
                    chunk_index: Tuple[int]) -> Tuple[int]:
    return tuple(s % c if ((i + 1) * c > s) else c
                 for s, c, i in zip(shape, chunks, chunk_index))


def get_array_chunk_indexes(num_chunks: Tuple[int]) -> Iterator[Tuple[int]]:
    return itertools.product(*tuple(map(range, map(int, num_chunks))))


def get_array_chunk_keys(array_name: str,
                         num_chunks: Tuple[int]) -> Iterator[str]:
    for chunk_index in get_array_chunk_indexes(num_chunks):
        yield format_array_chunk_key(array_name, chunk_index)


def format_array_chunk_key(array_name: str,
                           chunk_index: Tuple[int]) -> str:
    chunk_id = '.'.join(map(str, chunk_index))
    return f"{array_name}/{chunk_id}"


def dict_to_bytes(d: Dict):
    return str_to_bytes(json.dumps(d, indent=2))


def str_to_bytes(s: str):
    return bytes(s, encoding='utf-8')
