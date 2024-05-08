# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Union

import xarray as xr

from xcube.core.mldataset import FsMultiLevelDataset
from xcube.core.mldataset import FsMultiLevelDatasetError
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.subsampling import AGG_METHODS

# Note, we need the following reference to register the
# xarray property accessor
# noinspection PyUnresolvedReferences
from xcube.core.zarrstore import ZarrStoreHolder
from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonComplexSchema
from xcube.util.jsonschema import JsonIntegerSchema
from xcube.util.jsonschema import JsonNullSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema
from .dataset import DatasetZarrFsDataAccessor
from ... import DataStoreError
from ...datatype import DATASET_TYPE
from ...datatype import DataType
from ...datatype import MULTI_LEVEL_DATASET_TYPE


class MultiLevelDatasetLevelsFsDataAccessor(DatasetZarrFsDataAccessor):
    """Opener/writer extension name 'mldataset:levels:<protocol>'."""

    @classmethod
    def get_data_type(cls) -> DataType:
        return MULTI_LEVEL_DATASET_TYPE

    @classmethod
    def get_format_id(cls) -> str:
        return "levels"

    def open_data(self, data_id: str, **open_params) -> MultiLevelDataset:
        assert_instance(data_id, str, name="data_id")
        fs, fs_root, open_params = self.load_fs(open_params)
        try:
            return FsMultiLevelDataset(data_id, fs=fs, fs_root=fs_root, **open_params)
        except FsMultiLevelDatasetError as e:
            raise DataStoreError(f"{e}") from e

    def get_write_data_params_schema(self) -> JsonObjectSchema:
        schema = super().get_write_data_params_schema()  # creates deep copy
        # TODO: remove use_saved_levels, instead see #619
        schema.properties["use_saved_levels"] = JsonBooleanSchema(
            description="Whether to open an already saved level"
            " and downscale it then."
            " May be used to avoid computation of"
            " entire Dask graphs at each level.",
            default=False,
        )
        schema.properties["base_dataset_id"] = JsonStringSchema(
            description="If given, avoids writing the base dataset"
            ' at level 0. Instead a file "{data_id}/0.link"'
            " is created whose content is the given base dataset"
            " identifier.",
            nullable=True,
        )
        schema.properties["tile_size"] = JsonComplexSchema(
            one_of=[
                JsonIntegerSchema(minimum=1),
                JsonArraySchema(
                    items=[JsonIntegerSchema(minimum=1), JsonIntegerSchema(minimum=1)]
                ),
                JsonNullSchema(),
            ],
            description="Tile size to be used for all levels of the"
            " written multi-level dataset.",
        )
        schema.properties["num_levels"] = JsonIntegerSchema(
            description="Maximum number of levels to be written.",
            minimum=1,
            nullable=True,
        )
        schema.properties["agg_methods"] = JsonComplexSchema(
            one_of=[
                JsonStringSchema(enum=AGG_METHODS),
                JsonObjectSchema(
                    additional_properties=JsonStringSchema(enum=AGG_METHODS)
                ),
                JsonNullSchema(),
            ],
            description="Aggregation method for the pyramid levels."
            ' If not explicitly set, "first" is used for integer '
            ' variables and "mean" for floating point variables.'
            " If given as object, it is a mapping from variable "
            " name pattern to aggregation method. The pattern"
            " may include wildcard characters * and ?.",
        )
        return schema

    def write_data(
        self,
        data: MultiLevelDataset,
        data_id: str,
        replace: bool = False,
        **write_params,
    ) -> str:
        assert_instance(data, MultiLevelDataset, name="data")
        return self.write_data_generic(data, data_id, replace=replace, **write_params)

    def write_data_generic(
        self,
        data: Union[xr.Dataset, MultiLevelDataset],
        data_id: str,
        replace: bool = False,
        **write_params,
    ) -> str:
        assert_instance(data, (xr.Dataset, MultiLevelDataset), name="data")
        fs, fs_root, write_params = self.load_fs(write_params)
        base_dataset_id = write_params.pop("base_dataset_id", None)
        try:
            return FsMultiLevelDataset.write_dataset(
                data,
                data_id,
                fs=fs,
                fs_root=fs_root,
                replace=replace,
                base_dataset_path=base_dataset_id,
                **write_params,
            )
        except FsMultiLevelDatasetError as e:
            raise DataStoreError(f"{e}") from e


class DatasetLevelsFsDataAccessor(MultiLevelDatasetLevelsFsDataAccessor):
    """Opener/writer extension name 'dataset:levels:<protocol>'."""

    @classmethod
    def get_data_type(cls) -> DataType:
        return DATASET_TYPE

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        ml_dataset = super().open_data(data_id, **open_params)
        return ml_dataset.get_dataset(0)

    def write_data(
        self, data: xr.Dataset, data_id: str, replace: bool = False, **write_params
    ) -> str:
        assert_instance(data, xr.Dataset, name="data")
        return self.write_data_generic(data, data_id, replace=replace, **write_params)
