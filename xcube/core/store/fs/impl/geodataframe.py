# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from abc import abstractmethod

import geopandas as gpd
import pandas as pd

from xcube.util.assertions import assert_instance
from xcube.util.fspath import is_local_fs
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.temp import new_temp_file
from ..accessor import FsDataAccessor
from ...datatype import DataType
from ...datatype import GEO_DATA_FRAME_TYPE


class GeoDataFrameFsDataAccessor(FsDataAccessor):
    """Extension name: 'geodataframe:<format_id>:<protocol>'."""

    @classmethod
    def get_data_type(cls) -> DataType:
        return GEO_DATA_FRAME_TYPE

    @classmethod
    @abstractmethod
    def get_driver_name(cls) -> str:
        """Get the GeoDataFrame I/O driver name"""

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                storage_options=self.get_storage_options_schema(),
                # TODO: add more, see https://geopandas.org/io.html
            ),
        )

    def open_data(self, data_id: str, **open_params) -> gpd.GeoDataFrame:
        # TODO: implement me correctly,
        #  this is not valid for shapefile AND geojson
        fs, root, open_params = self.load_fs(open_params)
        is_local = is_local_fs(fs)
        if is_local:
            file_path = data_id
        else:
            _, file_path = new_temp_file()
            fs.get_file(data_id, file_path)
        return gpd.read_file(file_path, driver=self.get_driver_name(), **open_params)

    def get_write_data_params_schema(self) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                storage_options=self.get_storage_options_schema(),
                # TODO: add more, see https://geopandas.org/io.html
            ),
        )

    def write_data(self, data: gpd.GeoDataFrame, data_id: str, **write_params) -> str:
        # TODO: implement me correctly,
        #  this is not valid for shapefile AND geojson
        assert_instance(data, (gpd.GeoDataFrame, pd.DataFrame), "data")
        fs, root, write_params = self.load_fs(write_params)
        is_local = is_local_fs(fs)
        if is_local:
            file_path = data_id
        else:
            _, file_path = new_temp_file()
        data.to_file(file_path, driver=self.get_driver_name(), **write_params)
        if not is_local:
            fs.put_file(file_path, data_id)
        return data_id


class GeoDataFrameShapefileFsDataAccessor(GeoDataFrameFsDataAccessor):
    """Extension name: 'geodataframe:shapefile:<protocol>'."""

    @classmethod
    def get_format_id(cls) -> str:
        return "shapefile"

    @classmethod
    def get_driver_name(cls) -> str:
        return "ESRI Shapefile"


class GeoDataFrameGeoJsonFsDataAccessor(GeoDataFrameFsDataAccessor):
    """Extension name: 'geodataframe:geojson:<protocol>'."""

    @classmethod
    def get_format_id(cls) -> str:
        return "geojson"

    @classmethod
    def get_driver_name(cls) -> str:
        return "GeoJSON"
