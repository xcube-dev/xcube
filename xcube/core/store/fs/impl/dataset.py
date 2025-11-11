# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from abc import ABC
from typing import Optional

import xarray as xr
import zarr

# Note, we need the following reference to register the
# xarray property accessor
# noinspection PyUnresolvedReferences
from xcube.core.zarrstore import LoggingZarrStore
from xcube.util.assertions import assert_instance
from xcube.util.fspath import is_https_fs, is_local_fs
from xcube.util.jsonschema import (
    JsonArraySchema,
    JsonBooleanSchema,
    JsonIntegerSchema,
    JsonObjectSchema,
    JsonStringSchema,
)
from xcube.util.temp import new_temp_file

from ...datatype import DATASET_TYPE, DataType
from ...error import DataStoreError
from ..accessor import FsDataAccessor

ZARR_OPEN_DATA_PARAMS_SCHEMA = JsonObjectSchema(
    properties=dict(
        log_access=JsonBooleanSchema(default=False),
        cache_size=JsonIntegerSchema(
            minimum=0,
        ),
        group=JsonStringSchema(
            description="Group path. (a.k.a. path in zarr terminology.).",
            min_length=1,
        ),
        engine=JsonStringSchema(
            description="xarray backend name.",
            min_length=1,
        ),
        chunks=JsonObjectSchema(
            description="Optional chunk sizes along each dimension."
            ' Chunk size values may be None, "auto"'
            " or an integer value.",
            examples=[
                {"time": None, "lat": "auto", "lon": 90},
                {"time": 1, "y": 512, "x": 512},
            ],
            additional_properties=True,
        ),
        decode_cf=JsonBooleanSchema(
            description="Whether to decode these variables,"
            " assuming they were saved according to"
            " CF conventions.",
            default=True,
        ),
        mask_and_scale=JsonBooleanSchema(
            description="If True, replace array values equal"
            ' to attribute "_FillValue" with NaN. '
            ' Use "scale_factor" and "add_offset"'
            " attributes to compute actual values.",
            default=True,
        ),
        decode_times=JsonBooleanSchema(
            description="If True, decode times encoded in the"
            " standard NetCDF datetime format "
            "into datetime objects. Otherwise,"
            " leave them encoded as numbers.",
            default=True,
        ),
        decode_coords=JsonBooleanSchema(
            description='If True, decode the "coordinates"'
            " attribute to identify coordinates in "
            "the resulting dataset.",
            default=True,
        ),
        drop_variables=JsonArraySchema(
            items=JsonStringSchema(min_length=1),
        ),
        consolidated=JsonBooleanSchema(
            description="Whether to open the store using"
            " Zarr's consolidated metadata "
            "capability. Only works for stores that"
            " have already been consolidated.",
            default=False,
        ),
    ),
    required=[],
    # additional_properties=True because we want to allow passing arbitrary
    # parameters to xarray.open_dataset()
    additional_properties=True,
)

ZARR_WRITE_DATA_PARAMS_SCHEMA = JsonObjectSchema(
    properties=dict(
        group=JsonStringSchema(
            description="Group path. (a.k.a. path in zarr terminology.).",
            min_length=1,
        ),
        encoding=JsonObjectSchema(
            description="Nested dictionary with variable"
            " names as keys and dictionaries of"
            " variable specific encodings as values.",
            examples=[{"my_variable": {"dtype": "int16", "scale_factor": 0.1}}],
            additional_properties=True,
        ),
        consolidated=JsonBooleanSchema(
            description="If True (the default), consolidate all metadata"
            ' files ("**/.zarray", "**/.zattrs")'
            ' into a single top-level file ".zmetadata"',
            default=True,
        ),
        append_dim=JsonStringSchema(
            description="If set, the dimension on which the data will be appended.",
            min_length=1,
        ),
        replace=JsonBooleanSchema(
            description="If set, an existing dataset will be replaced without warning.",
        ),
    ),
    # additional_properties=True because we want to allow passing arbitrary
    # parameters to xarray.to_zarr()
    additional_properties=True,
)


class DatasetFsDataAccessor(FsDataAccessor, ABC):
    """Opener/writer extension name: 'dataset:<format>:<protocol>'."""

    @classmethod
    def get_data_type(cls) -> DataType:
        return DATASET_TYPE


class DatasetZarrFsDataAccessor(DatasetFsDataAccessor):
    """Opener/writer extension name: 'dataset:zarr:<protocol>'."""

    @classmethod
    def get_format_id(cls) -> str:
        return "zarr"

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        return self.add_storage_options_to_params_schema(ZARR_OPEN_DATA_PARAMS_SCHEMA)

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        assert_instance(data_id, str, name="data_id")
        fs, root, open_params = self.load_fs(open_params)
        engine = open_params.pop("engine", "zarr")
        cache_size = open_params.pop("cache_size", None)
        log_access = open_params.pop("log_access", None)
        consolidated = open_params.pop(
            "consolidated", fs.exists(f"{data_id}/.zmetadata")
        )
        zarr_store = fs.get_mapper(data_id)
        if isinstance(cache_size, int) and cache_size > 0:
            zarr_store = zarr.LRUStoreCache(zarr_store, max_size=cache_size)
        if log_access:
            zarr_store = LoggingZarrStore(zarr_store, name=f"zarr_store({data_id!r})")
        # TODO: test whether we really need to distinguish here as we know
        #   we are opening a zarr dataset, even without another backend.
        if engine == "zarr":
            try:
                dataset = xr.open_zarr(
                    zarr_store, consolidated=consolidated, **open_params
                )
            except ValueError as e:
                raise DataStoreError(f"Failed to open dataset {data_id!r}: {e}") from e
        else:
            try:
                dataset = xr.open_dataset(zarr_store, engine=engine, **open_params)
            except ValueError as e:
                raise DataStoreError(
                    f"Failed to open dataset {data_id!r} using engine {engine!r}: {e}"
                ) from e

        dataset.zarr_store.set(zarr_store)

        return dataset

    # noinspection PyMethodMayBeStatic
    def get_write_data_params_schema(self) -> JsonObjectSchema:
        return self.add_storage_options_to_params_schema(ZARR_WRITE_DATA_PARAMS_SCHEMA)

    def write_data(
        self, data: xr.Dataset, data_id: str, replace=False, **write_params
    ) -> str:
        assert_instance(data, xr.Dataset, name="data")
        assert_instance(data_id, str, name="data_id")
        fs, root, write_params = self.load_fs(write_params)
        zarr_store = fs.get_mapper(data_id, create=True)
        log_access = write_params.pop("log_access", None)
        if log_access:
            zarr_store = LoggingZarrStore(zarr_store, name=f"zarr_store({data_id!r})")
        consolidated = write_params.pop("consolidated", True)
        try:
            data.to_zarr(
                zarr_store,
                mode="w" if replace else None,
                consolidated=consolidated,
                **write_params,
            )
        except ValueError as e:
            raise DataStoreError(f"Failed to write dataset {data_id!r}: {e}") from e
        return data_id

    def delete_data(self, data_id: str, **delete_params):
        fs, root, delete_params = self.load_fs(delete_params)
        delete_params.pop("recursive", None)
        fs.delete(data_id, recursive=True, **delete_params)


NETCDF_OPEN_DATA_PARAMS_SCHEMA = JsonObjectSchema(
    properties=dict(
        # TODO: add more from xr.open_dataset()
    ),
    additional_properties=True,
)

NETCDF_WRITE_DATA_PARAMS_SCHEMA = JsonObjectSchema(
    properties=dict(
        # TODO: add more from ds.to_netcdf()
    ),
    additional_properties=True,
)


class DatasetNetcdfFsDataAccessor(DatasetFsDataAccessor):
    """Opener/writer extension name: 'dataset:netcdf:<protocol>'."""

    @classmethod
    def get_format_id(cls) -> str:
        return "netcdf"

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        return self.add_storage_options_to_params_schema(NETCDF_OPEN_DATA_PARAMS_SCHEMA)

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        assert_instance(data_id, str, name="data_id")
        fs, root, open_params = self.load_fs(open_params)

        # This doesn't yet work as expected with fsspec and netcdf:
        # engine = open_params.pop('engine', 'scipy')
        # with fs.open(data_id, 'rb') as file:
        #     return xr.open_dataset(file, engine=engine, **open_params)

        if is_local_fs(fs):
            file_path = data_id
        elif is_https_fs(fs):
            file_path = f"https://{data_id}#mode=bytes"
        else:
            _, file_path = new_temp_file(suffix=".nc")
            fs.get_file(data_id, file_path)
        engine = open_params.pop("engine", "netcdf4")
        return xr.open_dataset(file_path, engine=engine, **open_params)

    def get_write_data_params_schema(self) -> JsonObjectSchema:
        return self.add_storage_options_to_params_schema(
            NETCDF_WRITE_DATA_PARAMS_SCHEMA
        )

    def write_data(
        self, data: xr.Dataset, data_id: str, replace=False, **write_params
    ) -> str:
        assert_instance(data, xr.Dataset, name="data")
        assert_instance(data_id, str, name="data_id")
        fs, root, write_params = self.load_fs(write_params)
        if not replace and fs.exists(data_id):
            raise DataStoreError(f"Data resource {data_id} already exists")

        # This doesn't yet work as expected with fsspec and netcdf:
        # engine = write_params.pop('engine', 'scipy')
        # with fs.open(data_id, 'wb') as file:
        #     data.to_netcdf(file, engine=engine, **write_params)

        is_local = is_local_fs(fs)
        if is_local:
            file_path = data_id
        else:
            _, file_path = new_temp_file(suffix=".nc")
        engine = write_params.pop("engine", "netcdf4")
        data.to_netcdf(file_path, engine=engine, **write_params)
        if not is_local:
            fs.put_file(file_path, data_id)
        return data_id
