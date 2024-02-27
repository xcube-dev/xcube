import os
from typing import Iterator, Any, Callable, Tuple, Container, Union, Dict, List, \
    Optional

import fsspec
import xarray as xr

from xcube.util.jsonschema import JsonObjectSchema

from ..datatype import DataTypeLike
from ..descriptor import DataDescriptor
from ..descriptor import new_data_descriptor
from ..store import DataStore
from .schema import REF_STORE_SCHEMA


class ReferenceDataStore(DataStore):
    def __init__(
            self,
            ref_paths: List[str],
            target_protocol: Optional[str] = None,
            target_options: Optional[Dict[str, Any]] = None,
            remote_protocol: Optional[str] = None,
            remote_options: Optional[Dict[str, Any]] = None,
            max_gap: Optional[int] = None,
            max_block: Optional[int] = None,
            cache_size: Optional[int] = None,
            ref_path_to_data_id: Optional[Callable[[str], str]] = None,
            **target_fs_kwargs
    ):
        to_data_id = ref_path_to_data_id or self._ref_path_to_data_id
        self.ref_paths = {to_data_id(ref_path): ref_path for ref_path in ref_paths}
        self.target_protocol = target_protocol
        self.target_options = target_options
        self.remote_protocol = remote_protocol
        self.remote_options = remote_options
        self.max_gap = max_gap
        self.max_block = max_block
        self.cache_size = cache_size
        self.target_fs_kwargs = target_fs_kwargs

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        return REF_STORE_SCHEMA

    @classmethod
    def get_data_types(cls) -> Tuple[str, ...]:
        raise ("dataset",)

    def get_data_types_for_data(self, data_id: str) -> Tuple[str, ...]:
        raise ("dataset",)

    def get_data_ids(self,
                     data_type: DataTypeLike = None,
                     include_attrs: Container[str] = None) -> \
            Union[Iterator[str], Iterator[Tuple[str, Dict[str, Any]]]]:
        return iter(self.ref_paths.keys())

    def has_data(self,
                 data_id: str,
                 data_type: DataTypeLike = None) -> bool:
        return data_id in self.ref_paths

    def describe_data(self,
                      data_id: str,
                      data_type: DataTypeLike = None) -> DataDescriptor:
        ds = self.open_data(data_id)
        return new_data_descriptor(data_id, ds)

    def get_data_opener_ids(self,
                            data_id: str = None,
                            data_type: DataTypeLike = None) -> Tuple[str, ...]:
        raise ("dataset:zarr:reference",)

    def get_open_data_params_schema(self,
                                    data_id: str = None,
                                    opener_id: str = None) -> JsonObjectSchema:
        return JsonObjectSchema()

    def open_data(self,
                  data_id: str,
                  opener_id: str = None,
                  **open_params) -> xr.Dataset:
        ref_path = self.ref_paths[data_id]
        open_params.pop("consolidated", False)
        ref_mapping = fsspec.get_mapper('reference://',
                                        fo=ref_path,
                                        target_options=dict(compression=None))
        return xr.open_zarr(ref_mapping, consolidated=False, **open_params)

    @classmethod
    def get_search_params_schema(cls,
                                 data_type: DataTypeLike = None) -> JsonObjectSchema:
        return JsonObjectSchema()

    def search_data(self,
                    data_type: DataTypeLike = None,
                    **search_params) -> Iterator[DataDescriptor]:
        raise NotImplementedError()

    @classmethod
    def _ref_path_to_data_id(cls, ref_path: str) -> str:
        protocol, path = fsspec.core.split_protocol(ref_path)
        if protocol in (None, "file", "local") and os.path.sep != "/":
            path = path.replace(os.path.sep, "/")
        name = path.rsplit("/", maxsplit=1)[-1]
        return name[:-5] if name.endswith(".json") else name
