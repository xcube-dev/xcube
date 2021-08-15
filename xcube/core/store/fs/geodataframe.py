# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from abc import abstractmethod, ABC

import geopandas as gpd
import pandas as pd

from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonObjectSchema
from .common import FsDataAccessor


class GeoDataFrameFsDataAccessor(FsDataAccessor, ABC):
    """
    Extension name: "geodataframe:<format_id>:<fs_protocol>"
    """

    @classmethod
    def get_type_specifier(cls) -> str:
        return 'geodataframe'

    @classmethod
    @abstractmethod
    def get_driver_name(cls) -> str:
        """Get the GeoDataFrame I/O driver name"""

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                # TODO: add more, see https://geopandas.org/io.html
                fs_params=self.get_fs_params_schema(),
            ),
        )

    def open_data(self, data_id: str, **open_params) -> gpd.GeoDataFrame:
        # TODO: implement me correctly
        return gpd.read_file(data_id, driver=self.get_driver_name(), **open_params)

    def get_write_data_params_schema(self) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                # TODO: add more, see https://geopandas.org/io.html
                fs_params=self.get_fs_params_schema(),
            ),
        )

    def write_data(self, data: gpd.GeoDataFrame, data_id: str, **write_params):
        # TODO: implement me correctly
        assert_instance(data, (gpd.GeoDataFrame, pd.DataFrame), 'data')
        data.to_file(data_id, driver=self.get_driver_name(), **write_params)


class GeoDataFrameShapefileFsDataAccessor(GeoDataFrameFsDataAccessor, ABC):
    """
    Extension name: "geodataframe:shapefile:<fs_protocol>"
    """

    @classmethod
    def get_format_id(cls) -> str:
        return 'shapefile'

    @classmethod
    def get_driver_name(cls) -> str:
        return 'ESRI Shapefile'


# class GeoDataFrameShapefileFileFsDataAccessor(FileFsAccessor,
#                                             GeoDataFrameShapefileDataAccessor):
#     """
#     Opener/writer extension name: "geodataframe:shapefile:file"
#     """
#
#
# class GeoDataFrameShapefileS3FsDataAccessor(S3FsAccessor,
#                                           GeoDataFrameShapefileDataAccessor):
#     """
#     Opener/writer extension name: "geodataframe:shapefile:s3"
#     """
#
#
# class GeoDataFrameShapefileMemoryFsDataAccessor(MemoryFsAccessor,
#                                               GeoDataFrameShapefileDataAccessor):
#     """
#     Opener/writer extension name: "geodataframe:shapefile:memory"
#     """


class GeoDataFrameGeoJsonFsDataAccessor(GeoDataFrameFsDataAccessor, ABC):
    """
    Extension name: "geodataframe:geojson:<fs_protocol>"
    """

    @classmethod
    def get_format_id(cls) -> str:
        return 'geojson'

    @classmethod
    def get_driver_name(cls) -> str:
        return 'GeoJSON'

# class GeoDataFrameGeoJsonFileFsDataAccessor(FileFsAccessor,
#                                           GeoDataFrameGeoJsonDataAccessor):
#     """
#     Opener/writer extension name: "geodataframe:geojson:file"
#     """
#
#
# class GeoDataFrameGeoJsonS3FsDataAccessor(S3FsAccessor,
#                                         GeoDataFrameGeoJsonDataAccessor):
#     """
#     Opener/writer extension name: "geodataframe:geojson:s3"
#     """
#
#
# class GeoDataFrameGeoJsonMemoryFsDataAccessor(MemoryFsAccessor,
#                                             GeoDataFrameGeoJsonDataAccessor):
#     """
#     Opener/writer extension name: "geodataframe:geojson:memory"
#     """
