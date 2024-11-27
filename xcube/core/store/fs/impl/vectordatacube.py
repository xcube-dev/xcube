# Copyright (c) 2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import xarray as xr
import xvec

from xcube.core.mldataset.abc import VectorDataCube
from xcube.util.assertions import assert_instance
from ...datatype import DataType
from ...datatype import VECTOR_DATA_CUBE_TYPE

from .dataset import DatasetZarrFsDataAccessor
from .dataset import DatasetNetcdfFsDataAccessor


class VectorDataCubeZarrFsDataAccessor(DatasetZarrFsDataAccessor):
    """Opener/writer extension name: 'vectordatacube:zarr:<protocol>'."""

    @classmethod
    def get_data_type(cls) -> DataType:
        return VECTOR_DATA_CUBE_TYPE

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        dataset = super().open_data(data_id, **open_params)
        dataset = dataset.xvec.decode_cf()
        dataset = VectorDataCube.from_dataset(dataset)
        return dataset

    def write_data(
        self, data: xr.Dataset, data_id: str, replace=False, **write_params
    ) -> str:
        assert_instance(data, xr.Dataset, name="data")
        data_id = super().write_data(
            data.xvec.encode_cf(), data_id, replace, **write_params
        )
        return data_id


class VectorDataCubeNetcdfFsDataAccessor(DatasetNetcdfFsDataAccessor):
    """Opener/writer extension name: 'vectordatacube:netcdf:<protocol>'."""

    @classmethod
    def get_data_type(cls) -> DataType:
        return VECTOR_DATA_CUBE_TYPE

    def open_data(self, data_id: str, **open_params) -> VectorDataCube:
        dataset = super().open_data(data_id, **open_params)
        return dataset.xvec.decode_cf()

    def write_data(
        self, data: xr.Dataset, data_id: str, replace=False, **write_params
    ) -> str:
        assert_instance(data, xr.Dataset, name="data")
        data_id = super().write_data(
            data.xvec.encode_cf(), data_id, replace, **write_params
        )
        return data_id
