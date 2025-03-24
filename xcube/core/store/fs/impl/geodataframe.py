# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from abc import abstractmethod

import geopandas as gpd
import pandas as pd
import simplekml

from xcube.util.assertions import assert_instance
from xcube.util.fspath import is_local_fs
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.temp import new_temp_file

from ... import DataStoreError
from ...datatype import GEO_DATA_FRAME_TYPE, DataType
from ..accessor import FsDataAccessor


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
        replace = write_params.pop("replace", False)
        if is_local:
            file_path = data_id
            if not replace and fs.exists(file_path):
                raise DataStoreError(f"Data '{data_id}' already exists.")
        else:
            _, file_path = new_temp_file()
        data.to_file(file_path, driver=self.get_driver_name(), **write_params)
        if not is_local:
            mode = "overwrite" if replace else "create"
            fs.put_file(file_path, data_id, mode=mode)
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


class GeoDataFrameKmlFsDataAccessor(GeoDataFrameFsDataAccessor):
    """Extension name: 'geodataframe:kml:<protocol>'."""

    @classmethod
    def get_format_id(cls) -> str:
        return "kml"

    @classmethod
    def get_driver_name(cls) -> str:
        return "KML"

    def open_data(self, data_id: str, **open_params) -> gpd.GeoDataFrame:
        gdf = super().open_data(data_id, **open_params)
        kml_nan_columns = [
            "Name", "description", "timestamp", "begin", "end", "altitudeMode",
            "drawOrder", "icon"
        ]
        kml_number_columns = {
            "tessellate": -1,
            "extrude": 0,
            "visibility": -1,
        }
        for col in gdf.columns:
            if ((col in kml_nan_columns and pd.isna(gdf[col]).all()) or
                (col in kml_number_columns.keys() and
                 len(gdf[col].unique()) == 1 and
                 gdf[col].unique()[0] == kml_number_columns[col])):
                del gdf[col]
                continue
            if col not in ["geometry"]:
                try:
                    gdf[col] = pd.to_numeric(gdf[col])
                except ValueError:
                    if gdf[col].str.lower().isin(["true", "false"]).all():
                        gdf[col] = gdf[col].map({"true": True, "false": False})
                    else:
                        gdf[col] = gdf[col].astype(str)
        return gdf

    def write_data(self, data: gpd.GeoDataFrame, data_id: str, **write_params) -> str:
        assert_instance(data, (gpd.GeoDataFrame, pd.DataFrame), "data")
        fs, root, write_params = self.load_fs(write_params)
        is_local = is_local_fs(fs)
        replace = write_params.pop("replace", False)
        if is_local:
            file_path = data_id
            if not replace and fs.exists(file_path):
                raise DataStoreError(f"Data '{data_id}' already exists.")
        else:
            _, file_path = new_temp_file()

        kml = simplekml.Kml()

        for _, row in data.iterrows():
            geom = row.geometry

            if geom.geom_type == "Point":
                entry = kml.newpoint(coords=[(geom.x, geom.y)])
            elif geom.geom_type == "LineString":
                entry = kml.newlinestring(coords=list(geom.coords))
            elif geom.geom_type == "Polygon":
                entry = kml.newpolygon(outerboundaryis=list(geom.exterior.coords))
            else:
                continue
            if geom.geom_type in ["Point", "LineString", "Polygon"]:
                for col in data.columns:
                    if col != "geometry":
                        if isinstance((row[col]), bool):
                            entry.extendeddata.newdata(col, str(row[col]))
                        else:
                            entry.extendeddata.newdata(col, row[col])
        kml.save(file_path)
        if not is_local:
            mode = "overwrite" if replace else "create"
            fs.put_file(file_path, data_id, mode=mode)
        return data_id
